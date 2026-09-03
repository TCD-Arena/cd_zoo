import numpy as np
import pandas as pd
import networkx as nx


def run_ocse(data_sample, cfg):
    """
    oCSE (Optimal Causation Entropy) information-theoretic causal discovery for
    time series (Slote, Fish & Bollt, arXiv:2601.13365).

    An information-theoretic algorithm (training-free, classical — no neural
    network, no pretrained weights) that discovers causal networks from time
    series data using conditional mutual information (CMI). It applies a
    two-phase approach: forward selection iteratively adds predictors that
    maximise conditional mutual information, followed by backward elimination
    that removes predictors losing significance when conditioned on others.
    The causal rule: variable X causes Y if knowing X(t) significantly improves
    prediction of Y(t+1), even when controlling for all other relevant
    variables. Supports Gaussian, kNN, KDE, geometric-kNN, and Poisson
    entropic estimators.

    Implementation: wraps ``causationentropy.discover_network`` (``pip install
    causationentropy`` — MIT, Python 3.12 + numpy <2 compatible, all deps
    already satisfied). The function takes a ``(T, n_vars)`` DataFrame (or
    ndarray transposed to that) and returns a ``networkx.MultiDiGraph`` whose
    directed edges ``(cause, effect)`` carry ``cmi`` (conditional mutual
    information = edge strength), ``p_value``, and ``lag`` attributes. The
    wrapper converts this graph into a continuous ``[effect, cause, lag]``
    score matrix scored by CMI (higher = stronger causal link), split into
    lagged (``[:, :, 1:]``) and instantaneous (``[:, :, 0]``) channels matching
    cd_zoo's convention. Edges with lag 0 populate the instantaneous channel;
    edges with lag ≥ 1 populate the corresponding lag slice.

    Note: the algorithm uses permutation testing (``n_shuffles`` iterations per
    edge) and can be computationally intensive; the default ``n_shuffles`` is
    reduced from the package default for speed.

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        Time series. DataFrame is ``(T, n_vars)``; ndarray is ``(n_vars, T)``
        (cd_zoo convention).
    cfg : OmegaConf
        ``max_lag`` (default ``3``), ``method`` (default ``'standard'``; one of
        ``'standard'``, ``'alternative'``, ``'information_lasso'``, ``'lasso'``),
        ``information`` (default ``'gaussian'``; one of ``'gaussian'``, ``'knn'``,
        ``'kde'``, ``'geometric_knn'``, ``'poisson'``), ``n_shuffles`` (default
        ``50``; permutation-test iterations), ``alpha_forward`` (default
        ``0.05``), ``alpha_backward`` (default ``0.05``).
    """
    from causationentropy import discover_network

    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        df = pd.DataFrame(data_sample.T)
        n_vars = data_sample.shape[0]
    else:
        df = data_sample
        n_vars = data_sample.shape[1]

    max_lag = cfg.max_lag
    n_shuffles = getattr(cfg, "n_shuffles", 50)
    method = getattr(cfg, "method", "standard")
    information = getattr(cfg, "information", "gaussian")
    alpha_forward = getattr(cfg, "alpha_forward", 0.05)
    alpha_backward = getattr(cfg, "alpha_backward", 0.05)

    try:
        G = discover_network(
            df,
            method=method,
            information=information,
            max_lag=max_lag,
            alpha_forward=alpha_forward,
            alpha_backward=alpha_backward,
            n_shuffles=n_shuffles,
            n_jobs=1,
        )

        # Build [effect, cause, lag] score matrix, scored by CMI.
        pred = np.zeros((n_vars, n_vars, max_lag + 1))
        # Map node labels to integer indices.
        node_list = list(df.columns) if hasattr(df, "columns") else list(range(n_vars))
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        for u, v, data in G.edges(data=True):
            cause = node_to_idx.get(u)
            effect = node_to_idx.get(v)
            lag = int(data.get("lag", 0))
            cmi = float(data.get("cmi", 0.0))
            if cause is None or effect is None:
                continue
            if lag > max_lag:
                lag = max_lag
            pred[effect, cause, lag] = max(pred[effect, cause, lag], cmi)
    except Exception as e:
        print(f"oCSE failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars, max_lag + 1))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred[:, :, 1:], pred[:, :, 0]
