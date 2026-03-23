import numpy as np
import lingam

def run_varlingam(data_sample, cfg):

    if isinstance(data_sample, np.ndarray):
        data_sample = data_sample.T

    model = lingam.VARLiNGAM(lags=cfg.max_lag, criterion=cfg.criterion, prune=cfg.prune)
    # bricks for very long time series so we have to shorten.
    try: #in rare occasions the fitting fails.
        model.fit(data_sample)
        # remove instant as its not available
        pred = np.transpose(model.adjacency_matrices_, axes=[1,2,0]) 
    except Exception as e:
        print(f"Fit failed and predicting zeros: {e}")
        pred = np.zeros((data_sample.shape[1], data_sample.shape[1], cfg.max_lag+1))

    if np.isnan(np.array(pred)).sum() > 0:
        print("ISSUE detected, prediction set to 0") 
        pred = np.zeros(pred.shape)

    pred = np.abs(pred)
    return pred[:,:,1:], pred[:,:,0]
