import hydra
from omegaconf import DictConfig
import pickle
import os

from causalrivers.tools.tools import (
    load_joint_samples,
    standard_preprocessing,
)


# Follow explanation in the README. This script requires CausalRivers to be installed and the data to be prepared with prepare_causal_rivers.py.

@hydra.main(version_base=None, config_path="config", config_name="prepare_causal_rivers.yaml")
def main(cfg: DictConfig):
    
    
    print(cfg)
    print("Loading data...")
    # First, we load the full dataset from path and preprocess according to config..
    test_data, test_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing if cfg.dt_preprocess else None
    )
    
    
    test_labels = [x.values for x in test_labels]
    test_data = [x.values  for x in test_data]
    
    # run this extra for fpcmci to not run into numpy version issues (use the fpcmci environment.)
    
    # make save folder if it does not exist
    os.makedirs(cfg.save_path, exist_ok=True)
    
    
    pickle.dump((test_data, test_labels), open(cfg.save_path + "/release_random_5_tcd_arena.p", "wb"))  
    
    

if __name__ == "__main__":
    main()
