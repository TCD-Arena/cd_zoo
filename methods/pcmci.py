import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from hydra.utils import instantiate
import pandas as pd

def run_pcmci(data_sample,cfg):
    c_test  = instantiate(cfg.ci_test)

    try:
        if isinstance(data_sample, pd.DataFrame):
            dataframe = pp.DataFrame(data_sample.values,
                                    datatime = np.arange(len(data_sample)), 
                                    var_names=data_sample.columns,
                                    data_type=np.zeros_like(data_sample.values))
        else:
            dataframe = pp.DataFrame(data_sample.T,
                            datatime = np.arange(data_sample.T.shape[0]), 
                            var_names=np.arange(data_sample.T.shape[1]),
                            data_type=np.zeros_like(data_sample.T))
            
        pcmci = PCMCI(
            dataframe=dataframe, 
            cond_ind_test=c_test,
            verbosity=1)
        

        # for regression ci we set pc alpha to 0.01 because model selection is not implemented
        # in tigramite for this ci test.
        if "RegressionCI" in cfg.ci_test._target_ or "FCIT" in cfg.ci_test._target_:
            pc_alpha = 0.01 # standard value for PCMCI
        else:
            pc_alpha = None


        pcmci.verbosity = 0
        results = pcmci.run_pcmci(tau_max=cfg.max_lag, tau_min=1, pc_alpha=pc_alpha)
        pred = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')

        pred = results["p_matrix"]
        # remove instant link
        pred = pred[:,:,1:]
        pred = np.swapaxes(pred,0,1)
        # reverse as these are p values.
        pred = 1 - pred
        if np.isnan(pred).sum() > 0:
            print("ISSUE detected, prediction set to 0") 
            pred = np.zeros(pred.shape)
        return pred, None

    except Exception as e:
        print("Error in PCMCI: ", e)
        zero_pred =  np.zeros((data_sample.shape[0], data_sample.shape[0], cfg.max_lag))
        print(zero_pred.shape)
        return zero_pred, None

