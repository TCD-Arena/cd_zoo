import numpy as np
import pandas as pd
import torch

# Arrow's pretrained checkpoint is auto-fetched from HuggingFace on first use;
# cache the loaded model at module level and reuse it across samples (same
# pattern as AVICI / CDFM).
_MODEL_CACHE = {}


def _get_model(cfg):
    """Load (once) and cache the pretrained Arrow predictor."""
    if "arrow" not in _MODEL_CACHE:
        from arrowfm import ArrowPredictor
        print("[Arrow] loading pretrained model from HuggingFace")
        _MODEL_CACHE["arrow"] = ArrowPredictor()
    return _MODEL_CACHE["arrow"]


def run_arrow(data_sample, cfg):
    """
    Arrow (a foundation model for causal discovery) for i.i.d. observational
    data (Thompson, 2026).

    A zero-shot foundation model that predicts edge-existence probabilities and
    node-order scores, then decodes them into a directed acyclic graph. Given a
    dataset ``(N, p)`` it returns an adjacency matrix ``(p, p)``. Inputs are
    standardized across rows internally before inference.

    Arrow's ``predict_parameters`` returns ``p_hat`` edge probabilities in
    ``[cause, effect]`` orientation — i.e. ``p_hat[i, j]`` is the probability
    of the directed edge ``i -> j`` (i causes j). This follows directly from
    the architecture: ``p_hat[i,j] = a[i,j] * m[i,j]`` where ``a`` is the
    symmetric edge-existence probability and ``m[i,j] = sigmoid(s[i]-s[j])``
    is the probability that node ``i`` precedes node ``j`` in topological
    order — so a high ``p_hat[i,j]`` means ``i`` causes ``j``.
    cd_zoo scores in ``[effect, cause]`` orientation, so the prediction is
    **transposed** (``.T``) before scoring.
    (Verified empirically on the iid_nongaussian test set with known random
    DAGs: transposed gives AUROC 1.000 on every sample; as-is gives ~0.25,
    i.e. worse than chance — the signature of a transposition error. The
    ``decode_dag`` decoder uses the same ``[cause, effect]`` indexing:
    ``adj_hat[i,j] = edge & (i precedes j)`` = edge ``i -> j``.)

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        i.i.d. observations. DataFrame is ``(N_obs, n_vars)``; ndarray is
        ``(n_vars, N_obs)`` (cd_zoo convention).
    cfg : OmegaConf
        (no hyperparameters; the model is loaded from the released checkpoint)
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values

    n_vars = X.shape[1]
    try:
        model = _get_model(cfg)
        x = torch.from_numpy(np.ascontiguousarray(X)).float()
        p_hat, _ = model.predict_parameters(x)  # [cause, effect], in [0, 1]
        pred = p_hat.numpy().T  # transpose to [effect, cause] for cd_zoo
    except Exception as e:
        print(f"Arrow failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred, None  # No lags / no separate instant graph: single summary.
