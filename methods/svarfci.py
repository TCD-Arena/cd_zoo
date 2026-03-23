import numpy as np
from tigramite import data_processing as pp
from methods.svar_tools import SVARFCI
from hydra.utils import instantiate
import pandas as pd
import datetime
from stopit import threading_timeoutable as timeoutable
# Adapted from https://github.com/jakobrunge/neurips2020
def run_svarfci(data_sample, cfg):
    c_test = instantiate(cfg.ci_test)
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
    try: # catch internal errors in extreme cases (e.g. very short time series with high lag)
        
        
        svarfci = SVARFCI(
            dataframe=dataframe, 
            cond_ind_test=c_test
        )

        @timeoutable()
        def run_with_timeout():
            _, svarfci.run_svarfci(tau_max=cfg.max_lag)
            return True
        # occasionally there is a bug in SVARFCI that freezes the process forever (likely some backend dies silently) for very short ts.
        # We are catching this by putting the run function in a timeout wrapper and return 0 if the time is hit.
        if data_sample.shape[1] < 10:
            print("Working with short ts, applying timeout wrapper.")
            res = run_with_timeout(timeout=30) 
            print(res)
            if not res:
                print("SVAR-FCI timeout hit, returning 0 prediction")
                pred = np.zeros((data_sample.shape[0], data_sample.shape[0], cfg.max_lag))
                return pred[:, :, 1:], pred[:, :, 0]
        else:
            _ = svarfci.run_svarfci(tau_max=cfg.max_lag)
    

        pmat = svarfci._dict_to_matrix(svarfci.pval_max, svarfci.tau_max, svarfci.N)
        # reverse as these are p values.
        # remove instant link
        pred = np.swapaxes(pmat, 0, 1)
        # reverse as these are p values.
        pred = 1 - pred
        if np.isnan(pred).sum() > 0:
            print("ISSUE detected, prediction set to 0")
            pred = np.zeros(pred.shape)
        return pred[:, :, 1:], pred[:, :, 0]
    except Exception as e:
        print(e)
        pred = np.zeros((data_sample.shape[0], data_sample.shape[0], cfg.max_lag))
        return pred[:, :, 1:], pred[:, :, 0]

