import numpy as np
import pandas as pd
import lingam


def run_lingam(data_sample, cfg):
    """
    DirectLiNGAM for i.i.d. observational data (no time lags).

    Estimates a linear non-Gaussian acyclic model (LiNGAM) from a single
    i.i.d. sample and returns the absolute value of the structural-coefficient
    matrix ``B`` as the per-edge prediction score. ``B`` is stored in
    companion form ``B[effect, cause]`` (coefficient of the cause in the
    effect's structural equation), which matches cd_zoo's ``[effect, cause]``
    convention — no transpose is applied.

    Note: LiNGAM is fully identifiable (recovers the exact DAG, not just the
    Markov-equivalence class) only when the noise is non-Gaussian. With
    (near-)Gaussian noise the orientation of some edges is not identifiable
    and may be estimated arbitrarily.
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values

    n_vars = X.shape[1]
    try:
        model = lingam.DirectLiNGAM(random_state=cfg.random_state)
        model.fit(X)
        pred = np.abs(model.adjacency_matrix_)  # [effect, cause]
    except Exception as e:
        print(f"DirectLiNGAM failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred, None  # No lags / no separate instant graph: single summary.
