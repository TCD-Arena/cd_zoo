import pandas as pd
import numpy as np

# hotfix for numpy version that is too modern
try:
    import mxnet as mx
except Exception as e:
    np.bool = np.bool_
    np.float = np.float64
    import mxnet as mx

from methods.cdmi_tools import train_deep_ar_estimator as train_estimator
from methods.cdmi_tools import eval_deep_ar_estimator as eval_func
from methods.cdmi_tools import cdmi

# Adapted from CDMI_light repository.
def run_cdmi(data_sample, cfg):
    np.random.seed(1)
    mx.random.seed(2)

    if isinstance(data_sample, np.ndarray):
        data_sample = pd.DataFrame(data_sample.T)
        
    training_length = cfg.training_length
    training_data = data_sample.iloc[:training_length]
    num_windows = cfg.num_windows

    M = train_estimator(training_data,cfg) 
    # careful! depending on the metric smaller is more causal (p-value)
    pvals_stack  = cdmi(
        data_sample, M,eval_func,training_length, num_windows, cfg
    )
    res = 1- pvals_stack # to map this properly to p-values.

    # remove batch dimension here.
    if np.isnan(np.array(res)).sum() > 0:
        print("ISSUE detected, prediction set to 0") 
        res = np.zeros(res.shape)
    return res, None # No instant links predictable.




