"""Run cd_zoo causal-discovery methods on `causal-benchmarks` datasets.

This is the `causal-benchmarks` analogue of ``run_methods_on_causal_rivers.py``.
Because ``causal_benchmarks`` is pip-installable and returns ready-to-use
``CausalDataset`` objects (``.data`` DataFrame + ``.true_graph`` adjacency),
**no separate preparation/export step is needed** — data is loaded in-process
and fed straight into cd_zoo's :func:`tools.tools.benchmarking` /
:func:`tools.scoring_tools.score`.

Ground-truth orientation
------------------------
``causal-benchmarks`` stores summary graphs as ``adj[i,j]==1`` ⟺ *i causes j*
(``[cause, effect]``). cd_zoo methods return predictions as
``pred[effect, cause, lag]`` and score against labels in the same
``[effect, cause]`` convention, so we **transpose** each ground truth
(``true_graph.T``) before scoring. (Verified empirically: VAR on ``var_sl``
gives AUROC 1.0 with the transpose, ~0.44 without.)

cd_zoo includes both **time-series** methods (VAR, PCMCI+, DYNOTEARS, VarLiNGAM,
``lcm``, ``sdci``, …) and **i.i.d.** methods (``lingam``, ``pc``, ``ges``, ``ica_lingam``, ``dagma``, ``avici``, ``cdfm``, ``arrow``, ``tabcausal``).
Pointing a *time-series* method at an i.i.d. benchmark is not meaningful;
conversely, the i.i.d. methods are the right choice for i.i.d. benchmarks
(``domain == "iid"``).

Examples
--------
    # one method on one benchmark entry
    python run_methods_on_causal_benchmarks.py dataset=var_sl method=var

    # sweep methods × datasets with Hydra multirun
    python run_methods_on_causal_benchmarks.py --multirun \
        dataset=netsim,var_sl,kuramoto,lorenz96 method=var,dynotears,pcmciplus,lcm,sdci
"""
import datetime
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import hydra

from tools.scoring_tools import score
from tools.tools import benchmarking
from tools.method_loader import load_cd_method

from causal_benchmarks import load_dataset


@hydra.main(version_base=None, config_path="config",
            config_name="predict_causal_benchmarks.yaml")
def main(cfg: DictConfig):
    start = datetime.datetime.now()
    print(f"Method: {cfg.method.name}  |  Dataset: {cfg.dataset}")

    cd_method = load_cd_method(cfg.method.name)

    # Load the benchmark entry (a list[CausalDataset]).
    datasets = load_dataset(cfg.dataset)

    # Keep datasets that have a ground-truth graph and 2-D (T x N) tabular data.
    usable = [ds for ds in datasets
              if ds.true_graph is not None and ds.data.ndim == 2]
    if not usable:
        raise ValueError(
            f"No usable (2-D, has-ground-truth) datasets in '{cfg.dataset}'. "
            "Pick a time-series entry with a ground-truth graph.")

    if any(ds.domain != "timeseries" for ds in usable):
        # i.i.d. methods (lingam/pc/avici) are meaningful on i.i.d. data;
        # time-series methods (var/pcmci+/dynotears/...) are not.
        iids = sorted({cfg.method.name} & {"lingam", "pc", "ges", "ica_lingam", "dagma", "avici", "cdfm", "arrow", "tabcausal"})
        if not iids:
            print("WARNING: some datasets are not time-series and the selected "
                  "method is a time-series method — results on i.i.d. data are "
                  "not meaningful. Use an i.i.d. method (lingam/pc/ges/ica_lingam/dagma/avici/cdfm/arrow/tabcausal) for "
                  "i.i.d. benchmarks.")

    # cd_zoo's benchmarking() stacks predictions, so each batch needs a constant
    # node count. Group datasets by n_nodes (collections like `tcs` /
    # `causal_audit` have variable N); each group is scored separately.
    groups: dict[int, list] = defaultdict(list)
    for ds in usable:
        groups[ds.n_nodes].append(ds)

    results = []
    for n, grp in sorted(groups.items()):
        samples = [ds.data for ds in grp]                      # DataFrame T x N
        if cfg.restrict_to_n_steps > 0:
            samples = [s.iloc[:cfg.restrict_to_n_steps] for s in samples]
        # labels in cd_zoo [effect, cause] convention (transpose of causal-benchmarks)
        labels = np.stack([(ds.true_graph != 0).T for ds in grp]).astype(bool)

        print(f"N={n}: {len(grp)} datasets, T={samples[0].shape[0]}")
        lagged_preds, instant_preds = benchmarking(samples, cfg, cd_method)

        out = score(
            labels,
            lagged_preds,
            instant_labs=None,                       # causal-benchmarks GT is summary-only
            instant_preds=instant_preds,
            remove_autoregressive_for_lagged=False,
            verbose=cfg.verbose,
            per_sample_metrics=cfg.per_sample_metrics,
        )
        out.columns = [f"{cfg.dataset}_N{n}"]
        results.append(out)

    final = pd.concat(results, axis=1)
    print(final)

    # ---- save ------------------------------------------------------------- #
    p = (Path(cfg.save_path) / "causal_benchmarks" / cfg.dataset
         / cfg.method.name / "best_run")
    p.mkdir(parents=True, exist_ok=True)

    run_time = datetime.datetime.now() - start
    final.loc["runtime"] = run_time
    final.loc["dataset"] = cfg.dataset
    final.loc["method"] = cfg.method.name
    final.to_csv(p / "scoring.csv")
    with open(p / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    if cfg.save_predictions:
        pickle.dump(lagged_preds, open(p / "preds.p", "wb"))

    print(f"Saved results to: {p}")
    print("Done", datetime.datetime.now() - start)


if __name__ == "__main__":
    main()
