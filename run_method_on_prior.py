"""Bridge: run any cd_zoo causal-discovery method on the linear-PRIOR dataset.

The prior dataset (``/home/datasets4/stein/prior/linear_new``) is a grid of
synthetic VAR/SEM samples generated from a linear prior over graphs — 91 cells
of the form ``nvars{N}_lags{L}_ts512`` (N∈2..16, L∈0..6), 5000 samples each
(455k total). Each sample is a time series ``X (N, T)`` with a ground-truth
**lagged** adjacency ``lagged_links (N, N, L)`` and an **instantaneous**
adjacency ``instant_links (N, N)``.

This script is a **reusable bridge**: point it at any cd_zoo method
(``var``, ``dynotears``, ``pcmciplus``, ``direct_crosscorr``, …) and it loads
the prior data and scores the method's predictions against the ground truth.

How it works
------------
1. **Loads data via the torch dataloader** ``cp/cp_2/components/dataloader.py``
   (``LinearPriorDataset``) — reuses its cell discovery, memory-mapped loading
   and whole-cell binarization (``any`` over the lag axis → summary graph).
2. **Per cell** (each cell has a constant ``n_vars``, so cd_zoo's
   ``benchmarking`` can stack predictions): subsamples ``--max-samples`` series,
   runs the method → lagged predictions ``[effect, cause, lag]``.
3. **Scores** with cd_zoo's ``score`` against the GT.

Ground-truth orientation
-------------------------
The prior's GT is stored in **companion form**: ``lagged_links[effect, cause,
lag]`` and ``instant_links[effect, cause]`` (source ``cause`` acts on target
``effect``) — see ``data_generator.py`` (``x_t = B x_t + Σ_l A_l x_{t-l}``,
``A_l = lagged_links[:,:,l]`` shape ``(k,k,p): effect, cause, lag``). This is
**exactly cd_zoo's ``[effect, cause]`` convention**, so **no transpose** is
needed (unlike ``causal-benchmarks``, whose ``[cause, effect]`` GT must be
transposed).

Caveats
-------
* cd_zoo methods are **lagged** time-series methods. VAR (default) returns
  lagged predictions only (``instant=None``), so the **instantaneous** GT is
  never scored by VAR — use a method that emits instant predictions
  (e.g. ``varlingam``) to score the instant graph.
* For cells with ``max_lags == 0`` the true summary (lagged) graph is empty
  (no time-delayed edges, only instantaneous ones). A lagged method like VAR
  has nothing correct to recover there, so AUROC is NaN (one class) — expected.
* Cells with true ``max_lags > --max-lag`` are under-lagged (VAR cannot see
  the higher-order true edges); the true ``max_lags`` is recorded per cell in
  the summary so the comparison stays interpretable.

Examples
--------
    # default: VAR (max_lag=3), 200 samples/cell, all 91 cells
    python run_method_on_prior.py

    # another method, more samples
    python run_method_on_prior.py --method dynotears --max-lag 3 --max-samples 500

    # restrict the grid
    python run_method_on_prior.py --n-vars 5 10 --max-lags-range 0 4
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# This script lives at the cd_zoo repository root, so cd_zoo's own ``tools/``
# package is importable directly (Python adds the script's directory to
# sys.path[0]). The prior dataloader lives in a *sibling* repository at
# ``cp/cp_2/components/dataloader.py`` and is imported there as
# ``from components.dataloader import ...`` (cp/cp_2 is its package root), so we
# put that root on sys.path rather than relying on a ``cp.`` prefix.
#
# IMPORTANT import order: cp/cp_2 has a top-level ``tools.py`` that collides with
# cd_zoo's ``tools/`` package. We must import cd_zoo's tools *first* (so
# ``sys.modules['tools']`` binds to cd_zoo's package) and only *then* add the
# cp/cp_2 root to the path. Because this script is *inside* cd_zoo, Python has
# already placed cd_zoo on sys.path[0], so the cd_zoo tools imports below bind
# the correct package before the cp/cp_2 path is inserted.
HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parent.parent          # cd_zoo → TCD-Arena → thesis_code
from tools.method_loader import load_cd_method  # noqa: E402
from tools.tools import benchmarking  # noqa: E402
from tools.scoring_tools import score  # noqa: E402

sys.path.insert(0, str(THESIS_ROOT / "cp" / "cp_2"))
from components.dataloader import LinearPriorDataset  # noqa: E402

DEFAULT_DATA = "/home/datasets4/stein/prior/linear_new"
DEFAULT_OUT = "/home/datasets4/stein/cd_benchmark_performances/prior_linear"
CDZOO_CFG_DIR = HERE / "config" / "method"


def build_method_cfg(method: str, max_lag: int) -> OmegaConf:
    """Load cd_zoo's method yaml (if any) and override max_lag; fall back to a
    minimal cfg so the bridge works for any registered method."""
    yml = CDZOO_CFG_DIR / f"{method}.yaml"
    if yml.exists():
        mc = OmegaConf.load(yml)
    else:
        mc = OmegaConf.create({"name": method})
    mc["name"] = method
    if "max_lag" in mc or max_lag > 0:
        mc["max_lag"] = max_lag
    return OmegaConf.create({"verbose": 0, "method": mc})


def run_cell(ds: LinearPriorDataset, cid: int, cell: dict, method: str,
             max_lag: int, max_samples: int, out_dir: Path) -> dict:
    """Run one method on one prior cell. Returns a summary row."""
    start = datetime.datetime.now()
    name = cell["name"]
    row = {"cell": name, "n_vars": cell["n_vars"], "true_max_lags": cell["max_lags"],
           "ts_n": cell["ts_n"], "method": method, "status": "pending",
           "n_samples": 0, "n_scored": 0, "runtime_s": 0.0,
           "lag_density": np.nan, "inst_density": np.nan, "error": ""}
    cfg = build_method_cfg(method, max_lag)
    try:
        X_mm, _, _ = ds._arrays(cid)                 # (m, N, T) memmap
        lag_sum, inst = ds._binary_graphs(cid)        # (m, N, N) bool each
        m = X_mm.shape[0]
        n = max_samples if (max_samples > 0 and max_samples < m) else m
        idx = np.arange(n)                            # deterministic first-n subsample
        X = np.array(X_mm[idx])                       # (n, N, T) materialised
        lag_labels = lag_sum[idx]                      # (n, N, N) binary
        inst_labels = inst[idx]                        # (n, N, N) binary
        row["n_samples"] = m
        row["n_scored"] = n
        row["lag_density"] = float(lag_labels.mean())
        row["inst_density"] = float(inst_labels.mean())

        cd_method = load_cd_method(method)
        lagged_preds, instant_preds = benchmarking(X, cfg, cd_method)

        out = score(lag_labels, lagged_preds,
                    instant_labs=inst_labels, instant_preds=instant_preds,
                    remove_autoregressive_for_lagged=False, verbose=0,
                    per_sample_metrics=False)
        # headline SG_max joint metrics (summary graph)
        for met in ["AUROC", "Max F1", "Acc", "SHD"]:
            key = f"{met} Joint"
            row[met] = float(out.loc[key, out.columns[0]]) if key in out.index else np.nan

        cell_out = out_dir / name / method
        cell_out.mkdir(parents=True, exist_ok=True)
        out.columns = [f"{name}"]
        out.loc["runtime"] = datetime.datetime.now() - start
        out.loc["cell"] = name
        out.loc["method"] = method
        out.to_csv(cell_out / "scoring.csv")
        OmegaConf.save(cfg, cell_out / "config.yaml")
        with open(cell_out / "cell_meta.json", "w") as f:
            json.dump(cell, f, indent=2)
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
    row["runtime_s"] = round((datetime.datetime.now() - start).total_seconds(), 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--method", default="var", help="cd_zoo method name")
    ap.add_argument("--max-lag", type=int, default=3, help="method max_lag")
    ap.add_argument("--max-samples", type=int, default=200,
                    help="samples per cell (-1 = all 5000)")
    ap.add_argument("--n-vars", type=int, nargs=2, default=None,
                    metavar=("MIN", "MAX"), help="restrict n_vars to [min,max]")
    ap.add_argument("--max-lags-range", type=int, nargs=2, default=None,
                    metavar=("MIN", "MAX"), help="restrict true max_lags to [min,max]")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "run.log", "w")

    def logp(*a, **kw):
        print(*a, **kw)
        print(*a, **{k: v for k, v in kw.items() if k != "flush"}, file=log, flush=True)

    nv = range(args.n_vars[0], args.n_vars[1] + 1) if args.n_vars else None
    ml = range(args.max_lags_range[0], args.max_lags_range[1] + 1) if args.max_lags_range else None
    ds = LinearPriorDataset(args.data_dir, n_vars=nv, max_lags=ml)
    cells = ds.cell_summary()
    logp(f"Bridge: method={args.method} max_lag={args.max_lag} "
         f"max_samples={args.max_samples} | {len(cells)} cells | "
         f"n_vars={sorted(set(c['n_vars'] for c in cells))} "
         f"max_lags={sorted(set(c['max_lags'] for c in cells))}")
    logp(f"out → {out_dir}")

    rows = []
    for i, cell in enumerate(cells, 1):
        cid = i - 1
        logp(f"[{i}/{len(cells)}] {cell['name']} (N={cell['n_vars']}, "
             f"lags={cell['max_lags']}) ...", end=" ", flush=True)
        r = run_cell(ds, cid, cell, args.method, args.max_lag,
                     args.max_samples, out_dir)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
        logp(f"→ {r['status']} | {r['n_scored']} scored | {r['runtime_s']}s | "
             f"AUROC={r.get('AUROC','NA')}")

    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False)
    cols = ["cell", "n_vars", "true_max_lags", "method", "status", "n_scored",
            "lag_density", "inst_density", "AUROC", "Max F1", "Acc", "SHD",
            "runtime_s"]
    summ = man[[c for c in cols if c in man.columns]].sort_values(
        ["n_vars", "true_max_lags"])
    summ.to_csv(out_dir / "summary_per_cell.csv", index=False)
    logp("\n=== SUMMARY (per cell, SG_max joint metrics) ===")
    logp(summ.to_string(index=False))
    logp(f"\nDone. Results in {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
