from causalnex.structure.dynotears import from_pandas_dynamic
import numpy as np
import pandas as pd


def run_dynotears(data_sample, cfg):
    if isinstance(data_sample, np.ndarray):
        data_sample = pd.DataFrame(data_sample.T)
    try: # catch internal errors in extreme cases (e.g. very short time series with high lag)

        # do not accidentally break the formatting later due to overlapiing namings
        data_sample.columns = ["Var" + str(x) for x in range(len(data_sample.columns))]
        sm = from_pandas_dynamic(
            data_sample.reset_index(drop=True),
            p=cfg.max_lag,
            lambda_w=cfg.lambda_w,
            lambda_a=cfg.lambda_a,
            max_iter=cfg.max_iter,
            h_tol=cfg.h_tol,
        )
        pred = reformat_dynotears(data_sample, sm, cfg.max_lag)
        return pred[:,:,1:], pred[:,:,0]
    except Exception as e:
        print(e)
        pred = np.zeros((len(data_sample.columns),len(data_sample.columns),cfg.max_lag))
        return pred[:,:,1:], pred[:,:,0]


def reformat_dynotears(data_sample, sm, max_lag):
    pred = list(sm.adjacency())
    # remove lag 0
    #pred = [x for x in pred if x[0].split("_")[-1] != "lag0"]
    result = np.zeros((len(data_sample.columns), len(data_sample.columns), max_lag+1))
    mapping = {x: n for n, x in enumerate(data_sample.columns)}

    # effect x cause x lag
    for x in pred:
        # cause
        lag = x[0][-1]
        cause = x[0].split("_")[0]
        # effect:
        for key in x[1].keys():
            effect = key.split("_")[0]
            value = x[1][key]["weight"]
            result[mapping[effect], mapping[cause], int(lag)] = value
    
    return np.abs(result) # We use the absolute values as prediction for a link to exist
