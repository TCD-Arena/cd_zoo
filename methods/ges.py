import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges

# GES returns a causal-learn CPDAG (GeneralGraph) with the same endpoint
# coding as PC, so the CPDAG-to-score conversion is identical. Reuse the
# helper defined in methods.pc rather than duplicating it.
from methods.pc import _cpdag_to_scores


def run_ges(data_sample, cfg):
    """
    GES (Greedy Equivalence Search) score-based causal discovery for i.i.d.
    observational data (Chickering 2002).

    A score-based algorithm that searches over the space of CPDAGs (Markov
    equivalence classes of DAGs) by greedily adding and then deleting edges to
    greedily maximise a decomposable score (here the BIC, by default). Unlike
    the constraint-based PC, GES does not require conditional-independence
    tests; it evaluates whole graphs by a score and is generally more robust
    with limited data. Like PC it returns a CPDAG: edges that are not
    identifiable from the score alone remain undirected.

    Implementation: wraps ``causallearn.search.ScoreBased.GES.ges`` (causal-learn
    is already a dependency of cd_zoo, used by the PC method). The function
    returns a dict whose ``'G'`` entry is a causal-learn ``GeneralGraph`` whose
    ``.graph`` attribute encodes the learned CPDAG with the standard endpoint
    convention (``-1`` tail, ``1`` arrowhead) — exactly the same format PC
    produces. The prediction is therefore derived with the same
    ``_cpdag_to_scores`` helper as PC: directed edges score ``1.0``,
    undirected edges ``0.5``, absent edges ``0.0``, in ``[effect, cause]``
    orientation (no transpose).

    Parameters
    ----------
    data_sample : pandas.DataFrame or np.ndarray
        i.i.d. observations. DataFrame is ``(N_obs, n_vars)``; ndarray is
        ``(n_vars, N_obs)`` (cd_zoo convention).
    cfg : OmegaConf
        ``score_func`` (default ``local_score_BIC``; alternatives include
        ``local_score_CV_general``, ``local_score_marginal_general`` for
        non-Gaussian / nonlinear likelihoods), ``maxP`` (default ``None`` =
        no cap on the number of parents).
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values

    n_vars = X.shape[1]
    try:
        score_func = getattr(cfg, "score_func", "local_score_BIC")
        maxP = getattr(cfg, "maxP", None)
        Record = ges(X, score_func=score_func, maxP=maxP)
        pred = _cpdag_to_scores(Record["G"].graph)  # [effect, cause]
    except Exception as e:
        print(f"GES failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred, None  # No lags / no separate instant graph: single summary.
