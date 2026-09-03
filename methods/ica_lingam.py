import numpy as np
import pandas as pd
import lingam


def run_ica_lingam(data_sample, cfg):
    """
    ICA-LiNGAM for i.i.d. observational data (Shimizu et al. 2006).

    Estimates a linear non-Gaussian acyclic model (LiNGAM) using
    Independent Component Analysis. It is the original LiNGAM estimator
    (predecessor of DirectLiNGAM): it first separates the data into
    independent components via ICA and then permutes them into a causal order.
    Like DirectLiNGAM it is fully identifiable only under non-Gaussian noise,
    but it can be more brittle for large variable counts and is best suited to
    smaller graphs.

    Implementation: wraps ``lingam.ICALiNGAM`` (the ``lingam`` package, already a
    dependency of cd_zoo via the DirectLiNGAM method). The API is identical to
    DirectLiNGAM — ``ICALiNGAM().fit(X)`` then ``.adjacency_matrix_`` — so the
    wrapper mirrors ``methods.lingam`` exactly. ``adjacency_matrix_`` is stored
    in companion form ``B[effect, cause]`` (coefficient of the cause in the
    effect's structural equation), matching cd_zoo's ``[effect, cause]``
    convention — no transpose is applied (same orientation as DirectLiNGAM).

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        i.i.d. observations. DataFrame is ``(N_obs, n_vars)``; ndarray is
        ``(n_vars, N_obs)`` (cd_zoo convention).
    cfg : OmegaConf
        ``random_state`` (default ``0``) and optional ``threshold`` (default
        ``0.0`` = no hard thresholding; scores are kept continuous as the
        absolute structural-coefficient magnitude).
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values

    n_vars = X.shape[1]
    try:
        model = lingam.ICALiNGAM(random_state=cfg.random_state)
        model.fit(X)
        pred = np.abs(model.adjacency_matrix_)  # [effect, cause]
    except Exception as e:
        print(f"ICA-LiNGAM failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred, None  # No lags / no separate instant graph: single summary.
