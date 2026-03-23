import datetime

import hydra
import pickle
import numpy as np
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

from tools.scoring_tools import score
from tools.tools import benchmarking
from tools.method_loader import load_cd_method


# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="predict_causal_rivers.yaml")
def main(cfg: DictConfig):
    start = datetime.datetime.now()
    print(cfg.method)

    # Load the causal discovery method
    cd_method = load_cd_method(cfg.method.name)


    path = "causal_rivers"
    test_data, test_labels = pickle.load(open(cfg.base_path + cfg.ds_path, "rb"))
    
    test_data = [x.T for x in test_data]

    print(test_data[0].shape)

    if cfg.restrict_to > -1:
        test_data = test_data[cfg.restrict_to : cfg.restrict_to + cfg.n_restrict]
        test_labels = test_labels[cfg.restrict_to : cfg.restrict_to + cfg.n_restrict]


    test_data = [x[:,:cfg.restrict_to_n_steps] for x in test_data]
    print("Specifics:")
    print(len(test_data), test_data[0].shape)
    lagged_preds, instant_preds = benchmarking(test_data, cfg, cd_method)
    
    
    out = score(
        np.array(test_labels),
        lagged_preds.max(axis=3),    # collapse to summary graph
        None,
        None,
        remove_autoregressive_for_lagged=True,
        per_sample_metrics=True
    )
    # We have labels and predictions.

    print(out)
    
    
    p = Path(cfg.save_path) / "rivers" / cfg.method.name /"best_run"
    
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    
    end_time = datetime.datetime.now()
    run_time = end_time - start

    out.loc["runtime"] = run_time
    out.loc["path"] = path
        
    print("Saving results to: ", )
    out.to_csv(p / "scoring.csv")

    with open(p / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
        
    if cfg.save_predictions:
        pickle.dump(lagged_preds, open(p / "preds.p", "wb"))
    print("Done")


if __name__ == "__main__":
    main()
