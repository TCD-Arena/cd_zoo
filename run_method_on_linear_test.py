"""Bridge: run any cd_zoo causal-discovery method on the ``linear_test`` prior datasets.

The ``linear_test`` datasets (``/home/datasets4/stein/prior/linear_test``) are 6
targeted time-series benchmarks generated with ``cp/cp_2/prior``. Each is a
single cell nested one level deep under a scenario name::

    linear_test/<scenario>/nvars{N}_lags{L}_ts{T}_inno{noise}/
        X.npy              (m, N, T)              float32   observed time series
        lagged_links.npy   (m, N, N, L)           float32   lagged coefficients [effect, cause, lag]
        instant_links.npy  (m, N, N)              float32   instantaneous coeffs  [effect, cause]
        meta.json          {n_vars, max_lags, ts_n, inno_noise, n_samples, ...}

The 6 scenarios are designed as edge-case stress tests:

    gauss_inst_unid  nvars8  lags0  gauss   instantaneous-only, unidentifiable (Gaussian noise)
    many_lags        nvars5  lags8  gauss   high lag order (8)
    no_links         nvars5  lags0  gauss   null model (no edges)
    nvars20          nvars20 lags3  uniform  larger graph
    only_inst        nvars5  lags0  gauss   instantaneous only
    only_lagged      nvars5  lags3  gauss   lagged only (no instant edges)

This script is a **reusable bridge**: point it at any cd_zoo method
(``var``, ``varlingam``, ``pcmciplus``, ``dynotears``, …) and it loads each
dataset, runs the method, and scores the predictions against the ground truth.

How it works
------------
1. **Discovers** the 6 datasets by walking the nested scenario structure.
2. **Per dataset**: loads ``X``, ``instant_links``, ``lagged_links`` directly
   via ``np.load`` (no torch dataloader needed — the nesting differs from
   ``linear_new``'s flat grid, so ``LinearPriorDataset`` can't be reused).
3. Builds binary labels: **lagged summary** = OR over the lag axis (collapses
   to ``(m, N, N)``; all-False when ``max_lags == 0``); **instant** = nonzero.
4. Runs the method via cd_zoo's ``benchmarking`` → lagged predictions
   ``[effect, cause, lag]`` + optional instant predictions.
5. **Scores** with cd_zoo's ``score`` against the GT.

Ground-truth orientation
------------------------
The prior's GT is stored in **companion form**: ``lagged_links[effect, cause,
lag]`` and ``instant_links[effect, cause]`` (source ``cause`` acts on target
``effect`` — see ``cp/cp_2/components/dataloader.py`` docstring: "entry [i, j]
denotes source variable j acting on target variable i (j -> i)"). This is
**exactly cd_zoo's ``[effect, cause]`` convention**, so **no transpose** is
needed. (Verified empirically: VAR on ``only_lagged`` → AUROC 0.992 with no
transpose, 0.593 transposed.)

Caveats
-------
* cd_zoo methods are **lagged** time-series methods. VAR (default) returns
  lagged predictions only (``instant=None``), so the **instantaneous** GT is
  never scored by VAR — use a method that emits instant predictions
  (e.g. ``varlingam``) to score the instant graph.
* For datasets with ``max_lags == 0`` (``no_links``, ``only_inst``,
  ``gauss_inst_unid``) the true lagged (summary) graph is empty — a lagged
  method like VAR has nothing correct to recover there, so AUROC is NaN (one
  class) — expected.
* For ``many_lags`` (true ``max_lags == 8``) run with ``--max-lag >= 8`` or the
  method is under-lagged (cannot see the higher-order true edges); the true
  ``max_lags`` is recorded per dataset in the summary so the comparison stays
  interpretable.

Examples
--------
    # default: VAR (max_lag=3), all 6 datasets, 100 samples each
    python run_method_on_linear_test.py

    # another method, more samples, higher max_lag
    python run_method_on_linear_test.py --method varlingam --max-lag 8 --max-samples 100

    # restrict to specific scenarios
    python run_method_on_linear_test.py --scenarios only_lagged many_lags
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
# sys.path[0]).
HERE = Path(__file__).resolve().parent
from tools.method_loader import load_cd_method  # noqa: E402
from tools.tools import benchmarking  # noqa: E402
from tools.scoring_tools import score  # noqa: E402

DEFAULT_DATA = "/home/datasets4/stein/prior/linear_test"
DEFAULT_OUT = "/home/datasets4/stein/cd_benchmark_performances/prior_linear_test"
CDZOO_CFG_DIR = HERE / "config" / "method"


def discover_datasets(data_dir: str) -> list[dict]:
    """Walk ``data_dir/<scenario>/<nvars...>/`` and return one dict per dataset.

    Each dict: ``{scenario, name, dir, n_vars, max_lags, ts_n, inno_noise,
    n_samples}``. Sorted by scenario name.
    """
    import os
    import re
    cell_re = re.compile(r"nvars(\d+)_lags(\d+)_ts(\d+)(?:_inno(\w+))?$")
    required = ("X.npy", "lagged_links.npy", "instant_links.npy", "meta.json")
    datasets = []
    for scenario in sorted(os.listdir(data_dir)):
        scen_dir = os.path.join(data_dir, scenario)
        if not os.path.isdir(scen_dir):
            continue
        for name in sorted(os.listdir(scen_dir)):
            m = cell_re.match(name)
            if not m:
                continue
            cell_dir = os.path.join(scen_dir, name)
            if not os.path.isdir(cell_dir):
                continue
            if any(not os.path.exists(os.path.join(cell_dir, f)) for f in required):
                continue
            with open(os.path.join(cell_dir, "meta.json")) as fh:
                meta = json.load(fh)
            datasets.append({
                "scenario": scenario,
                "name": name,
                "dir": cell_dir,
                "n_vars": int(m.group(1)),
                "max_lags": int(m.group(2)),
                "ts_n": int(m.group(3)),
                "inno_noise": m.group(4) if m.group(4) else meta.get("inno_noise"),
                "n_samples": int(meta["n_samples"]),
            })
    return datasets


def build_method_cfg(method: str, max_lag: int) -> OmegaConf:
    """Load cd_zoo's method yaml (if any) and override max_lag."""
    yml = CDZOO_CFG_DIR / f"{method}.yaml"
    if yml.exists():
        mc = OmegaConf.load(yml)
    else:
        mc = OmegaConf.create({"name": method})
    mc["name"] = method
    if "max_lag" in mc or max_lag > 0:
        mc["max_lag"] = max_lag
    return OmegaConf.create({"verbose": 0, "method": mc})


def load_binary_labels(cell_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X and build binary (lagged_summary, instant) labels.

    Returns ``(X, lagged_summary, instant)`` where:
      X              : (m, N, T) float — observed time series
      lagged_summary : (m, N, N) bool  — OR over the lag axis (all-False if max_lags==0)
      instant        : (m, N, N) bool  — nonzero instantaneous coefficients
    All in [effect, cause] orientation (cd_zoo convention, no transpose).
    """
    X = np.load(f"{cell_dir}/X.npy")               # (m, N, T)
    lagged = np.load(f"{cell_dir}/lagged_links.npy") # (m, N, N, L)
    inst = np.load(f"{cell_dir}/instant_links.npy")  # (m, N, N)
    if lagged.shape[-1] > 0:
        lagged_summary = (np.abs(lagged) != 0).any(axis=-1)  # (m, N, N)
    else:
        lagged_summary = np.zeros(lagged.shape[:-1], dtype=bool)
    instant = (np.abs(inst) != 0)
    return X, lagged_summary, instant


def run_dataset(ds: dict, method: str, max_lag: int, max_samples: int,
                out_dir: Path) -> dict:
    """Run one method on one dataset. Returns a summary row."""
    start = datetime.datetime.now()
    name = ds["scenario"]
    row = {"scenario": name, "cell": ds["name"], "n_vars": ds["n_vars"],
           "true_max_lags": ds["max_lags"], "ts_n": ds["ts_n"],
           "inno_noise": ds["inno_noise"], "method": method, "status": "pending",
           "n_samples": 0, "n_scored": 0, "runtime_s": 0.0,
           "lag_density": np.nan, "inst_density": np.nan, "error": ""}
    cfg = build_method_cfg(method, max_lag)
    try:
        X, lag_labels, inst_labels = load_binary_labels(ds["dir"])
        m = X.shape[0]
        n = max_samples if (max_samples > 0 and max_samples < m) else m
        idx = np.arange(n)
        X = X[idx]                       # (n, N, T)
        lag_labels = lag_labels[idx]      # (n, N, N)
        inst_labels = inst_labels[idx]    # (n, N, N)
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
        # instant metrics if available
        if "INST" in out.columns:
            for met in ["AUROC", "Max F1"]:
                key = f"{met} Joint"
                row[f"INST_{met}"] = float(out.loc[key, "INST"]) if key in out.index else np.nan
        else:
            row["INST_AUROC"] = np.nan

        cell_out = out_dir / name / method
        cell_out.mkdir(parents=True, exist_ok=True)
        out.columns = [f"{name}_{c}" for c in out.columns]
        out.loc["runtime"] = datetime.datetime.now() - start
        out.loc["scenario"] = name
        out.loc["method"] = method
        out.to_csv(cell_out / "scoring.csv")
        OmegaConf.save(cfg, cell_out / "config.yaml")
        with open(cell_out / "dataset_meta.json", "w") as f:
            json.dump(ds, f, indent=2)
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
    ap.add_argument("--max-samples", type=int, default=100,
                    help="samples per dataset (-1 = all)")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="restrict to specific scenario names")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "run.log", "w")

    def logp(*a, **kw):
        print(*a, **kw)
        print(*a, **{k: v for k, v in kw.items() if k != "flush"}, file=log, flush=True)

    datasets = discover_datasets(args.data_dir)
    if args.scenarios:
        want = set(args.scenarios)
        datasets = [d for d in datasets if d["scenario"] in want]
    if not datasets:
        logp(f"No datasets found under {args.data_dir}")
        return
    logp(f"Bridge: method={args.method} max_lag={args.max_lag} "
         f"max_samples={args.max_samples} | {len(datasets)} datasets")
    for d in datasets:
        logp(f"  {d['scenario']:18s} N={d['n_vars']} lags={d['max_lags']} "
             f"ts={d['ts_n']} noise={d['inno_noise']} samples={d['n_samples']}")
    logp(f"out → {out_dir}")

    rows = []
    for i, ds in enumerate(datasets, 1):
        logp(f"[{i}/{len(datasets)}] {ds['scenario']} (N={ds['n_vars']}, "
             f"lags={ds['max_lags']}) ...", end=" ", flush=True)
        r = run_dataset(ds, args.method, args.max_lag, args.max_samples, out_dir)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
        lag_auroc = r.get("AUROC", "NA")
        inst_auroc = r.get("INST_AUROC", "NA")
        logp(f"→ {r['status']} | {r['n_scored']} scored | {r['runtime_s']}s | "
             f"lag_AUROC={lag_auroc} inst_AUROC={inst_auroc}")

    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False)
    cols = ["scenario", "n_vars", "true_max_lags", "inno_noise", "method",
            "status", "n_scored", "lag_density", "inst_density",
            "AUROC", "Max F1", "Acc", "SHD", "INST_AUROC", "runtime_s"]
    summ = man[[c for c in cols if c in man.columns]]
    summ.to_csv(out_dir / "summary_per_dataset.csv", index=False)
    logp("\n=== SUMMARY (per dataset, SG_max joint metrics) ===")
    logp(summ.to_string(index=False))
    logp(f"\nDone. Results in {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
