import numpy as np
from tigramite import data_processing as pp
from tigramite.lpcmci import LPCMCI
from hydra.utils import instantiate
import pandas as pd


def run_lpcmci(data_sample, cfg):
    """
    LPCMCI (Latent PCMCI) constraint-based causal discovery for time series
    with latent confounders (Gerhardus & Runge, NeurIPS 2020).

    LPCMCI relaxes the causal-sufficiency assumption: unlike PCMCI / PCMCI+ it
    allows for **hidden/latent confounders** and outputs a partial ancestral
    graph (PAG). It is the only tigramite method besides SVAR-FCI / SVAR-RFCI
    that handles latent confounders. It identifies low CI-test effect size as
    the root cause of poor recall of FCI-type methods on autocorrelated series
    and recovers it by including causal parents in the conditioning sets early,
    during edge removal, with novel orientation rules that already operate
    during the skeleton phase.

    Implementation: wraps ``tigramite.lpcmci.LPCMCI`` (tigramite is already a
    dependency of cd_zoo, used by PCMCI / PCMCI+ / SVAR-FCI / SVAR-RFCI). The
    API mirrors the PCMCI+ wrapper exactly: construct a tigramite
    ``data_processing.DataFrame``, instantiate ``LPCMCI(dataframe,
    cond_ind_test)``, and call ``run_lpcmci(tau_max=...)``. The continuous
    edge-score matrix is derived from ``pval_max_matrix`` (the max p-value
    across all CI tests for each link), reversed as ``1 - p`` so that stronger
    links score higher — the **same convention as PCMCI+ / SVAR-FCI**.
    ``causalnex`` is NOT required (tigramite is independent of it). The output
    is split into lagged (``[:, :, 1:]``) and instantaneous (``[:, :, 0]``)
    channels, matching cd_zoo's ``[effect, cause, lag]`` convention.

    Note: LPCMCI outputs a PAG (which can contain bidirected ``<->`` edges
    indicating latent confounders and circle ``o`` marks indicating
    uncertainty). The p-value matrix used here provides continuous edge
    strengths regardless of orientation marks; orientation is encoded in the
    separate ``.graph`` PAG (not used for scoring, as the AUROC evaluation
    treats edges as undirected strength comparisons).

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        Time series. DataFrame is ``(T, n_vars)``; ndarray is ``(n_vars, T)``
        (cd_zoo convention).
    cfg : OmegaConf
        ``max_lag`` (default ``3``), ``pc_alpha`` (default ``0.05``), and
        ``ci_test`` (a Hydra-instantiated conditional-independence test, e.g.
        ``ParCorr``).
    """
    c_test = instantiate(cfg.ci_test)

    try:
        if isinstance(data_sample, pd.DataFrame):
            dataframe = pp.DataFrame(
                data_sample.values,
                datatime=np.arange(len(data_sample)),
                var_names=data_sample.columns,
                data_type=np.zeros_like(data_sample.values),
            )
        else:
            dataframe = pp.DataFrame(
                data_sample.T,
                datatime=np.arange(data_sample.T.shape[0]),
                var_names=np.arange(data_sample.T.shape[1]),
                data_type=np.zeros_like(data_sample.T),
            )

        lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=0)

        # for regression-based CI tests, pc_alpha model selection is not
        # implemented in tigramite; fall back to a standard value.
        if "RegressionCI" in cfg.ci_test._target_ or "FCIT" in cfg.ci_test._target_:
            pc_alpha = 0.01
        else:
            pc_alpha = cfg.pc_alpha

        lpcmci.run_lpcmci(
            tau_max=cfg.max_lag,
            tau_min=0,
            pc_alpha=pc_alpha,
        )

        pred = lpcmci.pval_max_matrix  # (N, N, tau_max+1)
        pred = np.swapaxes(pred, 0, 1)  # -> [effect, cause, lag]
        # reverse as these are p values (low p = strong link)
        pred = 1 - pred
        if np.isnan(pred).sum() > 0:
            print("ISSUE detected, prediction set to 0")
            pred = np.zeros(pred.shape)
        return pred[:, :, 1:], pred[:, :, 0]

    except Exception as e:
        print("Error in LPCMCI: ", e)
        n = data_sample.shape[1] if isinstance(data_sample, pd.DataFrame) else data_sample.shape[0]
        zero_pred = np.zeros((n, n, cfg.max_lag))
        return zero_pred, np.zeros((n, n))
