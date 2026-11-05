"""Run any i.i.d. causal-discovery method on a time series by time-unrolling.

The series is turned into a window matrix whose columns are the variables at
different time offsets,

    [X_{1,t} .. X_{D,t} | X_{1,t-1} .. X_{D,t-1} | ... | X_{1,t-L} .. X_{D,t-L}],
    t = L+1 .. T,

i.e. ``T - L`` rows and ``D * (L + 1)`` columns.  The wrapped i.i.d. method
(any cd_zoo method returning a summary graph ``pred[effect, cause]``) is run on
this matrix, and its graph over the unrolled columns is read back into cd_zoo's
time-series output format:

* lagged edges  ``X_{j,t-l} -> X_{i,t}``  ->  ``lagged[i, j, l-1]``
* instantaneous ``X_{j,t}   -> X_{i,t}``  ->  ``instant[i, j]`` (diagonal zeroed)

All other entries (edges into lagged columns, edges pointing backwards in
time) are discarded.  This grants the i.i.d. method the same temporal
orientation that every lagged time-series method gets for free.

Caveats (by design of the experiment): rows overlap and the series is
autocorrelated, so the i.i.d. assumption of the wrapped method is violated,
and the lagged copies of a variable are strongly dependent columns.

Config (``config/method/unrolled.yaml``)
----------------------------------------
``base``            name of the wrapped i.i.d. method (``avici``, ``arrow``,
                    ``cdfm``, ``tabcausal``, ``lingam``, ``ica_lingam``, ``pc``,
                    ``ges``, ``dagma``).
``max_lag``         number of lags ``L`` to unroll.
``standardize``     z-score every unrolled column before calling the base.
``instantaneous``   return the instantaneous channel (else ``None``).
``base_cfg``        overrides for the base method's own config; the base's
                    ``config/method/<base>.yaml`` is loaded automatically.

Usage::

    python benchmark.py method=unrolled method.base=avici method.max_lag=3
"""
import os

import numpy as np
from omegaconf import OmegaConf

from tools.method_loader import load_cd_method

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "method")
_BASE_CACHE = {}


def _base_method(cfg):
    """Load (once) the wrapped i.i.d. method and its merged config."""
    base = cfg.base
    overrides = cfg.get("base_cfg", None) or {}
    key = (base, str(OmegaConf.to_container(OmegaConf.create(overrides))))
    if key not in _BASE_CACHE:
        base_cfg = OmegaConf.load(os.path.join(_CONFIG_DIR, f"{base}.yaml"))
        base_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))
        _BASE_CACHE[key] = (load_cd_method(base), base_cfg)
    return _BASE_CACHE[key]


def unroll(X, max_lag):
    """(T, D) series -> (T - L, D * (L + 1)) window matrix, block l = lag l."""
    T = X.shape[0]
    return np.concatenate([X[max_lag - l:T - l] for l in range(max_lag + 1)], axis=1)


def fold(big, n_vars, max_lag):
    """Graph over unrolled columns [effect, cause] -> (lagged, instant)."""
    big = np.abs(np.asarray(big, dtype=float))
    rows = big[:n_vars]                                   # effects at time t
    instant = rows[:, :n_vars].copy()
    np.fill_diagonal(instant, 0.0)
    lagged = np.stack([rows[:, l * n_vars:(l + 1) * n_vars]
                       for l in range(1, max_lag + 1)], axis=2)
    return lagged, instant


def run_unrolled(data_sample, cfg):
    # cd_zoo convention: ndarray is (vars, obs), DataFrame is (obs, vars).
    if isinstance(data_sample, np.ndarray):
        X = data_sample.T
    else:
        X = data_sample.values
    X = X.astype(float)
    n_vars, L = X.shape[1], int(cfg.max_lag)

    W = unroll(X, L)
    if cfg.get("standardize", True):
        W = (W - W.mean(0)) / (W.std(0) + 1e-8)

    fn, base_cfg = _base_method(cfg)
    try:
        big, _ = fn(W.T, base_cfg)                         # (vars, obs) ndarray
        lagged, instant = fold(big, n_vars, L)
    except Exception as e:
        print(f"unrolled({cfg.base}) failed, predicting zeros: {e}")
        lagged, instant = np.zeros((n_vars, n_vars, L)), np.zeros((n_vars, n_vars))

    if np.isnan(lagged).any() or np.isnan(instant).any():
        print(f"NAN predicted by unrolled({cfg.base}). replacing with 0")
        lagged, instant = np.nan_to_num(lagged), np.nan_to_num(instant)

    return lagged, (instant if cfg.get("instantaneous", True) else None)
