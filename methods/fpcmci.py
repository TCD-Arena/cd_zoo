import numpy as np
#from tigramite import data_processing as pp
#from methods.candoit_tools import CAnDOIT, Data, ParCorr, CPLevel
from hydra.utils import instantiate
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.causal_discovery.tigramite.independence_tests.parcorr import ParCorr as CIParCorr
from causalflow.causal_discovery.tigramite.independence_tests.robust_parcorr import RobustParCorr   
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.CPrinter import CPLevel

## Adapted from https://github.com/lcastri/causalflow/tree/main/causalflow

def run_fpcmci(data_sample,cfg):
    n_vars = data_sample.shape[0]
    
    if cfg.ci_test == "parcorr":
        ci_test = CIParCorr()
    elif cfg.ci_test == "robust_parcorr":
        ci_test = RobustParCorr()
    else:
        raise ValueError(f"Unknown ci_test: {cfg.ci_test}")
    
    data_sample = Data(data_sample.T)
    fpcmci = FPCMCI(
                data_sample,
                min_lag=0,
                max_lag=cfg.max_lag,
                sel_method = TE(TEestimator.Gaussian), 
                val_condtest = ci_test,
                verbosity = CPLevel.NONE)
    # Occasionally we might run into internal errors, catch them here and predict 0.                
    try:
        pred = fpcmci.run()
        #return nan nan if all features are removed in the first step
        if pred == (None,None):
            pred = np.zeros((n_vars, n_vars, cfg.max_lag+1))
            
        else:  
            pval_matrix = pred.get_pval_matrix()
            skeleton = pred.get_skeleton()
            included_list = list(pred.get_Adj().keys())
            if np.isnan(pval_matrix).sum() > 0:
                    print("ISSUE detected, prediction set to 0") 
                    pval_matrix = np.zeros(pval_matrix.shape)
                    
                    
            # SOMETHING IS OFF WITH THE SKELETON FIX
            skeleton = (skeleton != 1)

            # return 0 for removed variables.
            n_lags = pval_matrix.shape[-1] # Usually 4, but safer to read from data

            # 1. Parse the strings "X_0", "X_5" into integers 0, 5
            # This creates a list of indices that exist in the DAG
            included_indices = [int(x.split('_')[1]) for x in included_list]
            included_indices.sort() # Ensure sorted order to match pval_matrix structure

            # 2. Initialize with Zeros
            # Your logic implies that if a var is missing, the row/col is 0. 
            # So we start with all zeros.
            full_matrix = np.zeros((n_vars, n_vars, n_lags))
            full_skeleton = np.zeros((n_vars, n_vars, n_lags)).astype(bool)

            # 3. Map the subset matrix into the full matrix
            # np.ix_ allows us to map the (k, k) pval_matrix into specific (k, k) coordinates of full_matrix
            if len(included_indices) > 0:
                full_matrix[np.ix_(included_indices, included_indices)] = pval_matrix 
                full_skeleton[np.ix_(included_indices, included_indices)] = ~skeleton 
                pred = full_matrix
                # as they are p values we flip and put 0 where there is no link.
                pred[~full_skeleton] = 1

                pred = 1 - pred
            else:
                # No connected variables returned.
                pred = np.zeros((n_vars, n_vars, n_lags))
    except Exception as e:
        print("Error in FPCMCI: ", e)
        zero_pred =  np.zeros((n_vars, n_vars, cfg.max_lag))
        print(zero_pred.shape)
        return zero_pred, np.zeros((n_vars, n_vars))

    return  pred[:,:,1:],pred[:,:,0]



