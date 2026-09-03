import numpy as np
import pandas as pd

# TabCausal's pretrained checkpoint is auto-fetched from HuggingFace on first
# use; cache the loaded predictor at module level and reuse it across samples
# (same pattern as AVICI / CDFM / Arrow).
_MODEL_CACHE = {}


def _get_model(cfg):
    """Load (once) and cache the pretrained TabCausal predictor from HuggingFace."""
    repo = getattr(cfg, "repo_id", "LAMDA-Tabular/TabCausal")
    ckpt = getattr(cfg, "checkpoint", "checkpoints/tabcausal-base.pt")
    device = getattr(cfg, "device", None)
    key = (repo, ckpt)
    if key not in _MODEL_CACHE:
        from huggingface_hub import hf_hub_download
        from tabcausal import TabCausalPredictor
        print(f"[TabCausal] loading pretrained model from HuggingFace: {repo} :: {ckpt}")
        path = hf_hub_download(repo_id=repo, filename=ckpt)
        _MODEL_CACHE[key] = TabCausalPredictor(path, device=device)
    return _MODEL_CACHE[key]


def run_tabcausal(data_sample, cfg):
    """
    TabCausal (a causal discovery foundation model for tabular data) for i.i.d.
    observational data (LAMDA, 2026).

    A zero-shot foundation model that, given purely observational data
    ``X (N, D)``, predicts directed edge probabilities ``P (D, D)`` in a single
    forward pass. It is trained on a large, diverse distribution of synthetic
    structural causal models spanning many graph families, mechanisms, and
    sample sizes, and generalises zero-shot to unseen node counts, sample
    sizes, and SCM families. Each variable is standardised within-graph before
    inference (handled internally by the predictor).

    The model's two-channel public input format ``[observations, variables, 2]``
    pairs each value with an intervention indicator. The cd_zoo i.i.d. contract
    feeds purely observational data, so the intervention channel is set to all
    zeros.

    TabCausal's ``predict_proba`` returns ``probabilities[i, j]`` = P(edge
    ``i -> j``) (i causes j), i.e. in ``[cause, effect]`` orientation. cd_zoo
    scores in the ``[effect, cause]`` convention, so the prediction is
    transposed (``probabilities.T``) before returning — the same convention as
    AVICI and CDFM. (Verified empirically: transposed gives AUROC 1.0 on a
    known linear collider-chain SCM, as-is gives ~0.52.)

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        i.i.d. observations. DataFrame is ``(N_obs, n_vars)``; ndarray is
        ``(n_vars, N_obs)`` (cd_zoo convention).
    cfg : OmegaConf
        ``repo_id`` (default ``LAMDA-Tabular/TabCausal``, the released
        HuggingFace repo), ``checkpoint`` (default
        ``checkpoints/tabcausal-base.pt``), ``device`` (default auto).
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values
    X = np.ascontiguousarray(X, dtype=np.float32)

    n_vars = X.shape[1]
    try:
        model = _get_model(cfg)
        # Build two-channel input [observations, variables, 2]: value + 0
        # intervention flag (purely observational data).
        x_in = np.stack([X, np.zeros_like(X)], axis=-1)
        probs = model.predict_proba(x_in)  # [cause, effect], in [0, 1]
        probs = np.asarray(probs)
        if probs.ndim == 3:  # batched -> take single example
            probs = probs[0]
        pred = probs.T  # -> [effect, cause]
    except Exception as e:
        print(f"TabCausal failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred, None  # No lags / no separate instant graph: single summary.
