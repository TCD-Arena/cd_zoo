"""Sweep the cd_zoo VAR method over every loadable time-series benchmark in the
`causal-benchmarks` registry.

For each benchmark entry this script:
  1. loads its ``CausalDataset`` list,
  2. keeps datasets with a 2-D ``data`` frame and a ground-truth graph,
  3. groups them by node count (cd_zoo's ``benchmarking`` stacks predictions, so
     each batch needs a constant N),
  4. runs VAR (``max_lag=3``, the cd_zoo default) via ``benchmarking``,
  5. scores against the ground truth (transposed to cd_zoo's [effect, cause]
     convention) with ``score`` (SG_max summary graph),
  6. writes ``scoring.csv`` + ``config.yaml`` per entry, and appends to a
     ``summary_per_group.csv`` / ``summary_per_dataset.csv``.

Results go to ``/home/datasets4/stein/cd_benchmark_performances``.

Ground-truth orientation: causal-benchmarks ``adj[i,j]=1`` ⟺ i causes j
([cause, effect]); cd_zoo scores [effect, cause], so we transpose each GT
before scoring (verified: VAR on ``var_sl`` → AUROC 1.0 with the transpose).

Each entry is wrapped in a hard timeout so one pathological series cannot stall
the whole sweep; failures are recorded in the manifest.
"""
from __future__ import annotations

import datetime
import signal
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# This script lives at the cd_zoo repository root, so cd_zoo's own ``tools/``
# package is importable directly (Python adds the script's directory to
# sys.path[0]).
from tools.method_loader import load_cd_method  # noqa: E402
from tools.tools import benchmarking  # noqa: E402
from tools.scoring_tools import score  # noqa: E402

import causal_benchmarks as cb  # noqa: E402

OUT = Path("/home/datasets4/stein/cd_benchmark_performances")
METHOD = "var"
MAX_LAG = 3
PER_ENTRY_TIMEOUT = 1800  # seconds (30 min) — generous; causal_audit (500 ds) ~1-2 min

# All 24 time-series collection entries that carry a ground truth. Each covers
# its sub-entries exactly once (e.g. `tcs` covers tcs_cp_* + tcs_fmri;
# `causal_audit` covers f1..f10). Two needed optional deps installed for this run
# (causalchamber, causalcompass); see NOTES below.
ENTRIES = [
    "aerosol_cloud", "causal_chambers", "causalcompass", "causalrivers",
    "cpcad", "ess_accp", "river_discharge_danube", "causaltime", "netsim",
    "tcs", "causal_audit", "causaldynamics_simple", "causaldynamics_coupled",
    "causaldynamics_climate", "kuramoto", "lorenz96", "nri", "tcd_arena",
    "timegraph", "var_lnl", "var_ml", "var_snl", "var_mnl", "var_xlnl",
]

CFG = OmegaConf.create({
    "verbose": 0,
    "method": {"name": METHOD, "max_lag": MAX_LAG,
               "absolute_coefficients": True, "base_on": "coefficients"},
})


class TimeoutError_(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError_("per-entry timeout")


def run_one(entry: str) -> dict:
    """Run VAR on one benchmark entry. Returns a manifest row dict."""
    start = datetime.datetime.now()
    out_dir = OUT / entry / METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {"dataset": entry, "method": METHOD, "status": "pending",
           "n_datasets": 0, "n_groups": 0, "n_scored": 0, "runtime_s": 0.0,
           "error": ""}
    try:
        datasets = cb.load_dataset(entry)
        usable = [ds for ds in datasets
                 if ds.true_graph is not None and ds.data.ndim == 2]
        row["n_datasets"] = len(usable)
        if not usable:
            row["status"] = "no_usable_data"
            return row
        groups: dict[int, list] = defaultdict(list)
        for ds in usable:
            groups[ds.n_nodes].append(ds)
        row["n_groups"] = len(groups)

        all_out = []
        group_rows = []
        for n, grp in sorted(groups.items()):
            samples = [ds.data for ds in grp]
            labels = np.stack([(ds.true_graph != 0).T for ds in grp]).astype(bool)
            lagged, instant = benchmarking(samples, CFG, load_cd_method(METHOD))
            out = score(labels, lagged, instant_labs=None, instant_preds=instant,
                        remove_autoregressive_for_lagged=False, verbose=0,
                        per_sample_metrics=True)
            out.columns = [f"{entry}_N{n}"]
            all_out.append(out)
            # headline SG_max joint metrics for this group
            g = {"dataset": entry, "n_nodes": n, "n_in_group": len(grp)}
            for met in ["AUROC", "Max F1", "Acc", "SHD"]:
                key = f"{met} Joint"
                g[met] = float(out.loc[key, out.columns[0]]) if key in out.index else np.nan
            group_rows.append(g)
            row["n_scored"] += len(grp)

        final = pd.concat(all_out, axis=1)
        final.loc["runtime"] = datetime.datetime.now() - start
        final.loc["dataset"] = entry
        final.loc["method"] = METHOD
        final.to_csv(out_dir / "scoring.csv")
        OmegaConf.save(CFG, out_dir / "config.yaml")
        pd.DataFrame(group_rows).to_csv(out_dir / "summary_per_group.csv", index=False)

        # weighted-mean headline across groups (weighted by n_in_group)
        gdf = pd.DataFrame(group_rows)
        w = gdf["n_in_group"].values.astype(float)
        for met in ["AUROC", "Max F1", "Acc", "SHD"]:
            vals = gdf[met].values.astype(float)
            mask = ~np.isnan(vals)
            row[f"wm_{met}"] = float(np.average(vals[mask], weights=w[mask])) if mask.any() else np.nan
        row["status"] = "ok"
    except TimeoutError_:
        row["status"] = "timeout"
        row["error"] = f"exceeded {PER_ENTRY_TIMEOUT}s"
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
    row["runtime_s"] = round((datetime.datetime.now() - start).total_seconds(), 1)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _alarm_handler)
    print(f"Sweeping VAR (max_lag={MAX_LAG}) over {len(ENTRIES)} TS benchmark "
          f"entries → {OUT}")
    manifest = []
    group_rows = []
    for i, entry in enumerate(ENTRIES, 1):
        print(f"\n[{i}/{len(ENTRIES)}] {entry} ...", flush=True)
        signal.alarm(PER_ENTRY_TIMEOUT)
        try:
            row = run_one(entry)
        finally:
            signal.alarm(0)
        manifest.append(row)
        # collect group rows if the run produced a group CSV
        gcsv = OUT / entry / METHOD / "summary_per_group.csv"
        if gcsv.exists():
            group_rows.append(pd.read_csv(gcsv))
        else:
            # failed entries still get an empty placeholder so concat works
            group_rows.append(pd.DataFrame([{"dataset": entry}]))
        # stream manifests so progress is saved even if interrupted
        pd.DataFrame(manifest).to_csv(OUT / "manifest.csv", index=False)
        print(f"    → {row['status']} | {row['n_scored']} scored | "
              f"{row['runtime_s']}s | wm_AUROC={row.get('wm_AUROC','NA')}", flush=True)

    man = pd.DataFrame(manifest)
    man.to_csv(OUT / "manifest.csv", index=False)
    if group_rows:
        pd.concat(group_rows, ignore_index=True).to_csv(
            OUT / "summary_per_group.csv", index=False)
    # per-dataset weighted-mean summary
    cols = ["dataset", "status", "n_datasets", "n_scored", "runtime_s",
            "wm_AUROC", "wm_Max F1", "wm_Acc", "wm_SHD"]
    summary = man[[c for c in cols if c in man.columns]].copy()
    summary.to_csv(OUT / "summary_per_dataset.csv", index=False)
    print("\n=== SUMMARY (per dataset, weighted-mean SG_max joint metrics) ===")
    print(summary.to_string(index=False))
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
