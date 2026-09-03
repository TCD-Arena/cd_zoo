import numpy as np
import pandas as pd
import avici

# AVICI's pretrained model is expensive to load (checkpoint download + JAX
# graph construction), so we cache it at module level and reuse it across
# samples instead of reloading per call.
_MODEL_CACHE = {}


def _get_model(cfg):
    """Load (once) and cache the pretrained AVICI model selected by cfg.download."""
    name = getattr(cfg, "download", "scm-v0")
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = avici.load_pretrained(download=name)
    return _MODEL_CACHE[name]


def run_avici(data_sample, cfg):
    """
    AVICI (Amortized Variational Inference for Causal Inference) for i.i.d.
    observational data (Lorch et al., NeurIPS 2022).

    Runs a single forward pass through a pretrained neural network that maps a
    data matrix directly to a matrix of per-edge posterior probabilities. By
    default the general-purpose ``scm-v0`` checkpoint is used (trained on a
    broad prior of linear & nonlinear SCMs with Gaussian / Laplace / Cauchy
    noise and up to 100 nodes).

    AVICI's output ``g_prob[i, j]`` is the probability of the edge ``i -> j``
    (i causes j), i.e. stored in ``[cause, effect]`` orientation. cd_zoo scores
    in the ``[effect, cause]`` convention, so the prediction is transposed
    (``g_prob.T``) before returning. (Verified empirically: transposed gives
    AUROC ~0.99 on a known linear-Laplace SCM, as-is gives ~0.31.)
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values

    n_vars = X.shape[1]
    try:
        model = _get_model(cfg)
        g_prob = model(x=X)  # [cause, effect], values in [0, 1]
        pred = np.asarray(g_prob).T  # -> [effect, cause]
    except Exception as e:
        print(f"AVICI failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred, None  # No lags / no separate instant graph: single summary.
