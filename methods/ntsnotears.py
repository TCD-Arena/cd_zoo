from sklearn import preprocessing
import numpy as np
import torch
import pandas as pd
from methods.ntsnotears_tools import set_random_seed, NTS_NOTEARS, train_NTS_NOTEARS

"""Adapted from https://github.com/xiangyu-sun-789/NTS-NOTEARS/tree/main/notears"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def reshape_adj(mat, N, max_lag):
    # mat is of the form:
    # ['X1(t-3)', 'X2(t-3)', 'X3(t-3)', 'X4(t-3)', 'X1(t-2)', 'X2(t-2)', 'X3(t-2)', 'X4(t-2)', 
    #  'X1(t-1)', 'X2(t-1)', 'X3(t-1)', 'X4(t-1)', 'X1(t)', 'X2(t)', 'X3(t)', 'X4(t)']
    # for both rows and columns
    # meaning we have to reformat it to 
    # effect x cause x lag
    # to do this, we extract the last set of columns 
    # we do this to ensure no link to the past
    interest_region = mat[:,-N:]

    # we now have to split it according to the lags
    # we start with the last lag and got to lag t
    lags = [np.moveaxis(interest_region[i*N:(i+1)*N,:], 1, 0) for i in range(max_lag+1)]
    lags.reverse()

    result = np.stack(lags, axis=-1)
    return result

# Adapted from https://github.com/xiangyu-sun-789/NTS-NOTEARS/tree/main/notears
def run_ntsnotears(data_sample, cfg):
    
    if isinstance(data_sample, pd.DataFrame):
        data_sample = data_sample.to_numpy()
    else:
        data_sample = data_sample.T
    T, N = data_sample.shape
    max_lag = cfg.max_lag
    seed = cfg.seed
    h_tol = cfg.h_tol #1e-60
    rho_max = cfg.rho_max #1e+16
    lambda1 = cfg.lambda1 #0.005
    lambda2 = cfg.lambda2 #0.01

    try: # catch internal errors in extreme cases (e.g. very short time series with high lag)

        set_random_seed(seed)

        variable_names_no_time = ['X{}'.format(j) for j in range(1, N + 1)]

        X = data_sample
        scaler = preprocessing.StandardScaler().fit(X)
        normalized_X = scaler.transform(X)
        #Transformed to warning from assert of original.
        if (normalized_X.std(axis=0).round(decimals=3) == 1).all():  # make sure all the variances are (very close to) 1
            print("Warning: Normalized features have zero variance.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # eliminates edges smaller than the threshhold
        # we take all edges and then later use the best threshold to not underestimate method performance
        w_threshold = 0.0
        prior_knowledge = None

        M = NTS_NOTEARS(dims=[N, 10, 1], bias=True, number_of_lags=max_lag,
                            prior_knowledge=prior_knowledge, variable_names_no_time=variable_names_no_time)

        W_est_full = train_NTS_NOTEARS(M, normalized_X, device=device, lambda1=lambda1, lambda2=lambda2,
                                        w_threshold=w_threshold, h_tol=h_tol, rho_max=rho_max, verbose=0)
        # reshape to our standard: effect x cause x lags
        out =  reshape_adj(W_est_full, N, max_lag)
        return out[:,:,1:], out[:,:, 0]
    except Exception as e:
        print(e)
        pred = np.zeros((data_sample.shape[1], data_sample.shape[1], cfg.max_lag))
        return pred[:, :, 1:], pred[:, :, 0]
