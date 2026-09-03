import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc


def _cpdag_to_scores(g):
    """Convert a causal-learn CPDAG (GeneralGraph.graph) into a continuous
    ``[effect, cause]`` edge-score matrix.

    causal-learn encodes edge marks in ``g[i, j]`` (the mark at node *i* on the
    edge i-j): ``-1`` is a tail, ``1`` is an arrowhead. Hence:
      * directed  i -> j  :  g[i, j] == -1 (tail at i) and g[j, i] == 1 (arrow at j)
      * undirected i - j   :  g[i, j] == -1 and g[j, i] == -1
      * no edge            :  g[i, j] == 0 and g[j, i] == 0

    Scoring (higher = stronger belief that ``cause -> effect``):
      * a correctly oriented directed edge i->j scores 1.0 at [effect=j, cause=i]
      * an undirected edge scores 0.5 in both directions (genuine uncertainty)
      * an absent edge scores 0.0
    """
    n = g.shape[0]
    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            tail_i = g[i, j] == -1   # mark at i is a tail
            arrow_i = g[i, j] == 1   # mark at i is an arrowhead
            tail_j = g[j, i] == -1
            arrow_j = g[j, i] == 1
            if tail_i and arrow_j:
                # i -> j  (cause i, effect j)
                scores[j, i] = 1.0
            elif tail_j and arrow_i:
                # j -> i  (cause j, effect i)
                scores[i, j] = 1.0
            elif tail_i and tail_j:
                # i - j  (undirected): uncertain about direction
                scores[i, j] = 0.5
                scores[j, i] = 0.5
            # else: no edge -> 0.0
    return scores


def run_pc(data_sample, cfg):
    """
    PC constraint-based causal discovery for i.i.d. observational data.

    Runs the PC algorithm (Spirtes, Glymour, Scheines 2000) via causal-learn
    and returns a continuous ``[effect, cause]`` score matrix derived from the
    learned CPDAG: directed edges score 1.0 in their orientation, undirected
    edges score 0.5 in both directions, absent edges score 0.0.

    The default conditional-independence test is ``fisherz`` (partial
    correlation), which assumes linear-Gaussian relationships. For nonlinear
    data use e.g. ``kci`` or ``rcit`` (slower). PC returns a CPDAG: edges that
    are not oriented by colliders / background knowledge remain undirected.
    """
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values

    n_vars = X.shape[1]
    try:
        cg = pc(
            X,
            alpha=cfg.alpha,
            indep_test=cfg.indep_test,
            stable=True,
            show_progress=False,
        )
        pred = _cpdag_to_scores(cg.G.graph)  # [effect, cause]
    except Exception as e:
        print(f"PC failed, predicting zeros: {e}")
        pred = np.zeros((n_vars, n_vars))

    if np.isnan(pred).sum() > 0:
        print("NAN predicted. replacing with 0")
        pred = np.zeros(pred.shape)

    return pred, None  # No lags / no separate instant graph: single summary.
