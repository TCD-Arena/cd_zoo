import os

import numpy as np
import pandas as pd
import torch

from lcm_pytorch.modules.lcm_module import LCMModule
from lcm_pytorch.utils.utils import lagged_batch_crosscorrelation

# ---------------------------------------------------------------------------
# Large Causal Models (Kougioulis et al., 2026) — amortized *temporal*
# causal discovery.  A single pretrained transformer maps a multivariate
# time series directly to a lagged adjacency tensor in one forward pass.
#
# Pretrained checkpoints are hosted on Google Drive (not bundled).  They are
# downloaded on first use via ``gdown`` and cached under ``cfg.cache_dir``.
# The model is expensive to load, so it is cached at module level and reused
# across samples (like the AVICI / CP wrappers).
# ---------------------------------------------------------------------------

# Architecture bounds of the released checkpoints.
_MAX_VAR = 12      # maximum number of input variables (padding columns otherwise)
_MAX_LAG = 3       # maximum causal lag supported
_MAX_SEQ_LEN = 500  # maximum number of timesteps (padding rows otherwise)

# Google-Drive file ids for the released checkpoints (README, Table "Pretrained Models").
_GDRIVE_IDS = {
    "2.5M": "1vjzKMpr7M_feQuFgZGSb5VWRJ_7psV0v",
    "9.4M": "1UCjLG4Hs6MKSJF_5G3MJmTlhGQVp3qx7",
    "12.2M": "1bLKASu085xBJ0oqWbhNObcDie94eZFzR",
    "24M": "10gARotO3pK-bYnvpESNBlIV94SqxLxmw",
}

_MODEL_CACHE = {}


def _checkpoint_path(cfg):
    cache_dir = getattr(cfg, "cache_dir", "cd_zoo/methods/lcm_weights")
    size = getattr(cfg, "model_size", "2.5M")
    return os.path.join(cache_dir, f"LCM_{size}.ckpt"), cache_dir, size


def _download_checkpoint(file_id, dest):
    """Download a Google-Drive checkpoint with gdown (only when missing)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    import gdown
    gdown.download(id=file_id, output=dest, quiet=False)


def _get_model(cfg):
    """Load (once) and cache the pretrained LCM selected by cfg.model_size."""
    dest, cache_dir, size = _checkpoint_path(cfg)
    key = (os.path.abspath(dest), size)
    if key not in _MODEL_CACHE:
        if not os.path.exists(dest):
            if size not in _GDRIVE_IDS:
                raise ValueError(
                    f"Unknown LCM model_size '{size}'. "
                    f"Available: {sorted(_GDRIVE_IDS)}"
                )
            print(f"[LCM] downloading checkpoint '{size}' -> {dest}")
            _download_checkpoint(_GDRIVE_IDS[size], dest)
        model = LCMModule.load_from_checkpoint(dest, map_location="cpu")
        model.model.to("cpu").eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def run_lcm(data_sample, cfg):
    """
    Large Causal Model (Kougioulis et al., 2026) for *temporal* causal
    discovery from observational time series.

    A pretrained transformer ingests a multivariate time series and outputs a
    lagged adjacency tensor ``pred[effect, cause, lag]`` of per-edge
    confidence scores in a single forward pass.  The released checkpoints
    support up to 12 variables, 3 lags and 500 timesteps.  Narrower inputs
    are padded with low-variance noise variables; shorter series are used
    as they are, and longer ones are averaged over 500-step windows
    (following the official evaluation code in ``lcm_pytorch``).

    Output orientation is ``[effect, cause, lag]`` which matches cd_zoo's
    convention directly (no transpose).  The model emits lags in *descending*
    order (index 0 = highest lag), so the lag axis is flipped to ascending
    order (index 0 = lag 1) to match cd_zoo — identical to the Causal
    Pretraining (``cp``) wrapper.  (Verified: AUROC 1.0 on the official
    3-variable illustrative example with known ground truth.)

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        Time series.  DataFrame is ``(T, n_vars)``; ndarray is
        ``(n_vars, T)`` (cd_zoo convention).
    cfg : OmegaConf
        ``model_size`` (one of ``2.5M/9.4M/12.2M/24M``, default ``2.5M``) and
        ``cache_dir`` (default ``cd_zoo/methods/lcm_weights``).
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values
    X = X.astype(np.float32)
    n_vars = X.shape[1]

    try:
        if n_vars > _MAX_VAR:
            raise ValueError(
                f"LCM supports at most {_MAX_VAR} variables, got {n_vars}."
            )

        model = _get_model(cfg)
        M = model.model

        # Normalisation to [0, 1] (as in the official evaluation code,
        # lcm_pytorch.utils.misc_utils).
        X = (X - X.min()) / (X.max() - X.min() + 1e-8)
        Xt = torch.from_numpy(X)

        # The official evaluation does NOT pad the time axis: shorter series
        # are fed as they are (padding them with noise to 500 steps reduced
        # AUROC from .64 to .49 on 250-step series). Series longer than the
        # 500-step context are split into consecutive 500-step windows whose
        # predictions are averaged (cfg.window_average, default True); with
        # window_average=False they are truncated to the first 500 steps, as
        # in the official code.
        if Xt.shape[0] > _MAX_SEQ_LEN and getattr(cfg, "window_average", True):
            n_win = Xt.shape[0] // _MAX_SEQ_LEN
            windows = [Xt[k * _MAX_SEQ_LEN:(k + 1) * _MAX_SEQ_LEN] for k in range(n_win)]
        else:
            windows = [Xt[:_MAX_SEQ_LEN]]

        preds = []
        for W in windows:
            # Pad variables up to the model's V_max with low-variance noise.
            var_dif = _MAX_VAR - W.shape[1]
            if var_dif > 0:
                W = torch.cat([W, torch.normal(0, 0.01, (W.shape[0], var_dif))], dim=1)
            with torch.no_grad():
                corr = lagged_batch_crosscorrelation(W.unsqueeze(0), _MAX_LAG)
                corr = torch.nan_to_num(corr, nan=0.0)
                preds.append(torch.sigmoid(M((W.unsqueeze(0), corr))).squeeze(0).numpy())
        pred = np.mean(preds, axis=0)

        # Remove self-loops: the diagonal of every lag slice.
        idx = np.arange(pred.shape[0])
        pred[idx, idx, :] = 0.0

        # Keep only the real (top-left) variables, discarding padding columns.
        pred = pred[:n_vars, :n_vars, :]

        # Flip the lag axis: model emits index 0 = highest lag, cd_zoo expects
        # index 0 = lag 1 (ascending).
        pred = np.flip(pred, axis=2)

    except Exception as e:
        print(f"LCM failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars, _MAX_LAG))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted by LCM. replacing with 0")
        pred = np.zeros((n_vars, n_vars, _MAX_LAG))

    return pred, None  # No separate instant channel.
