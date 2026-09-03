import numpy as np
import pandas as pd

# DAGMA's optimisation is deterministic given the data; cache nothing (no
# pretrained weights). The per-call cost is the continuous-optimisation run.


def run_dagma(data_sample, cfg):
    """
    DAGMA (DAGs via M-matrices and a log-determinant acyclicity
    characterization) continuous-optimization causal discovery for i.i.d.
    observational data (Bello, Aragam, Ravikumar 2022).

    A continuous-optimization method for learning a directed acyclic graph from
    observational data. Like NOTEARS it replaces the combinatorial acyclicity
    constraint with a smooth, differentiable surrogate, but DAGMA uses a
    log-determinant formulation ``h(W) = -log det(sI - W∘W) + d·log s`` (over
    M-matrices) that has better-conditioned gradients, is about an order of
    magnitude faster than NOTEARS, and is better at detecting large cycles. It
    fits a weighted adjacency matrix ``W`` by a path-following sequence of
    unconstrained problems solved with accelerated gradient descent.

    Implementation: wraps ``dagma.linear.DagmaLinear`` (``pip install dagma``
    --no-deps; all transitive deps — numpy, scipy, torch, tqdm, igraph — were
    already satisfied by the cd_zoo environment). The API is
    ``DagmaLinear(loss_type).fit(X, lambda1)`` → ``.W_est``, a weighted weight
    matrix. DAGMA follows the NOTEARS convention where the reconstruction loss is
    ``||X − X @ W||²``, i.e. ``x_j = Σ_k W[k, j] · x_k``; thus ``W[k, j]`` is the
    coefficient of cause ``x_k`` in the equation of effect ``x_j`` — i.e.
    ``W[cause, effect]``. cd_zoo scores in the ``[effect, cause]`` convention,
    so the prediction is **transposed** (``|W_est|.T``) before returning.
    (Note this is the opposite of DirectLiNGAM, whose ``adjacency_matrix_`` is
    already ``[effect, cause]``. Verified empirically: transposed gives AUROC
    1.0 on a known collider-chain SCM, as-is gives ~0.30.) Self-loops on the
    diagonal are zeroed.

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        i.i.d. observations. DataFrame is ``(N_obs, n_vars)``; ndarray is
        ``(n_vars, N_obs)`` (cd_zoo convention).
    cfg : OmegaConf
        ``loss_type`` (default ``l2`` for continuous data; use ``logistic``
        for binary), ``lambda1`` (default ``0.02``, the L1 sparsity penalty),
        ``w_threshold`` (default ``0.0``; if >0, hard-threshold small weights
        before taking the absolute value).
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values

    n_vars = X.shape[1]
    try:
        from dagma.linear import DagmaLinear

        loss_type = getattr(cfg, "loss_type", "l2")
        lambda1 = getattr(cfg, "lambda1", 0.02)
        w_threshold = getattr(cfg, "w_threshold", 0.0)

        model = DagmaLinear(loss_type=loss_type, verbose=False)
        # DAGMA hardcodes a tqdm progress bar via `from tqdm.auto import tqdm`
        # + `with tqdm(total=...) as pbar:`. Silence it by temporarily
        # replacing that module-level name with a no-op context manager.
        import sys
        _dl = sys.modules.get("dagma.linear")

        class _NoPbar:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def update(self, *a, **k): pass
            def close(self): pass

        _orig_tqdm = getattr(_dl, "tqdm", None) if _dl else None
        if _dl is not None:
            _dl.tqdm = _NoPbar
        try:
            W = model.fit(X, lambda1=lambda1, w_threshold=w_threshold)
        finally:
            if _dl is not None and _orig_tqdm is not None:
                _dl.tqdm = _orig_tqdm
        W = np.asarray(W)
        pred = np.abs(W).T  # [cause, effect] -> [effect, cause]
        np.fill_diagonal(pred, 0.0)
    except Exception as e:
        print(f"DAGMA failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred, None  # No lags / no separate instant graph: single summary.
