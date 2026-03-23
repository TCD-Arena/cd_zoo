import numpy as np
from omegaconf import DictConfig
import hydra
from tools.helpers import summary_transform,load_single_samples, load_joint_samples,basic_preprocess
from tools.tools import benchmarking
from tools.scoring_tools import score
from tools.method_loader import load_cd_method


"""
Example script for some basic data formats.
Also check out benchmark.py for more streamlined benchmarking along the lines of TCD-Arena.
"""

# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="example_run.yaml")
def main(cfg: DictConfig):

    print(cfg.method)
    
    # Load the causal discovery method
    cd_method = load_cd_method(cfg.method.name)
    
    
    # Data loading. #We have two model here
    if cfg.loading_mode == "single":
        print("Folder specified.Attempting single load.")
        test_data, test_labels = load_single_samples(cfg)
    elif cfg.loading_mode == "joint":
        print("File specified. Attempting joint load.")
        test_data, test_labels = load_joint_samples(
            cfg,
            index_col=cfg.index_col,        )
    else:
        print("LOADING MODE NOT IMPLEMENTED")
        return 0
    
    
    test_data = basic_preprocess(test_data, cfg.data_preprocess)
    
    y_lagged, y_instant = benchmarking(test_data, cfg, cd_method)

    if test_labels[0].ndim == 2 and y_lagged[0].ndim == 3:
        # reduce lag dimension according so config
        y_lagged = np.array([summary_transform(x, cfg) for x in y_lagged])
        

    test_labels = np.array(test_labels)
    
    out = score(test_labels,y_lagged, cfg)
    print(out)



if __name__ == "__main__":
    main()
