import numpy as np
from tigramite import data_processing as pp
from methods.svar_tools import SVARRFCI
from hydra.utils import instantiate
import pandas as pd
from stopit import threading_timeoutable as timeoutable
# Adapted from https://github.com/jakobrunge/neurips2020
def run_svarrfci(data_sample, cfg):
    c_test = instantiate(cfg.ci_test)

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
            
            
        svarrfci = SVARRFCI(
            dataframe=dataframe, 
            cond_ind_test=c_test
        )
        
        
        @timeoutable()
        def run_with_timeout():
            _ = svarrfci.run_svarrfci(tau_max=cfg.max_lag)
            return True
        # occasionally there is a bug in SVARRFCI that freezes the process forever (likely some backend dies silently) for very short ts.
        # We are catching this by putting the run function in a timeout wrapper and return 0 if the time is hit.
        print(data_sample.shape)
        if data_sample.shape[1] < 10:
            print("Working with short ts, applying timeout wrapper.")
            res = run_with_timeout(timeout=5) 
            if not res:
                print("SVAR-RFCI timeout hit, returning 0 prediction")
                pred = np.zeros((data_sample.shape[0], data_sample.shape[0], cfg.max_lag))
                return pred[:, :, 1:], pred[:, :, 0]
        else:
            _ = svarrfci.run_svarrfci(tau_max=cfg.max_lag) 
        

        pred = svarrfci._dict_to_matrix(svarrfci.pval_max, svarrfci.tau_max, svarrfci.N)
        # reverse as these are p values.
        # remove instant link
        pred = np.swapaxes(pred, 0, 1)
        # reverse as these are p values.
        pred = 1 - pred
        if np.isnan(pred).sum() > 0:
            print("ISSUE detected, prediction set to 0")
            pred = np.zeros(pred.shape)
        return pred[:, :, 1:], pred[:, :, 0]

    except Exception as e:
        print("Error in SVAR-RFCI: ", e)
        zero_pred =  np.zeros((data_sample.shape[0], data_sample.shape[0], cfg.max_lag))
        print(zero_pred.shape)
        return zero_pred, np.zeros((data_sample.shape[0], data_sample.shape[0]))


