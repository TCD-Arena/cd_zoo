import numpy as np
import pandas as pd


def corr_filter(out, corr_map):
    # keeps only the highest corr value
    print("using corr filter")
    peak = corr_map.max(axis=2)
    peak[~out] = 0
    out = out * (peak == np.expand_dims(
        peak.max(axis=0),axis=0).repeat(peak.shape[0],axis=0))
    return out

def min_distance_filter(out,m):
    print("Selecting closest distance link.")
    # Calc distance between elements
    c = c = np.abs(m - m.T)
    # Fill all elements that are not relevant
    np.fill_diagonal(c,None)
    c[~out] = np.nan
    # Get the closest distance that remains
    d = np.nanmin(c,axis=0)
    d = np.stack([d]*m.shape[0])
    # Check which element is the closest distance
    # Keep it as the only link.
    return (c == d)
    

def max_distance_filter(out, m, cfg):
    print("Selecting max distance link.")
    # select only elements that are the max distance in the column
    # Calcs the distance and selects either the max or the minimum, 
    # depending on direction.
    
    distances = m - m.T
    # mask everything not relevant
    distances[~out] = np.nan
    if cfg.reverse_physical:
        return distances ==  np.stack([np.nanmax(distances,axis=0)]*m.shape[0])
    else:
        return distances ==  np.stack([np.nanmin(distances,axis=0)]*m.shape[0])

def calc_lagged_cross_corr(df, max_lag=500, grain=1):
    """
    return Crosscorrelation items for each variables with each variable for each lag. 

    Return format: Stable, moved, lag
    Importantly: If the peak is above the max_lag, 
    the cause likely goes from  stable to moved and the other way arround.
    """
    product = []
    for column in df.columns:
        stack = []
        for x in range(-max_lag,max_lag+1, grain):
            stack.append(df.shift(-x).corrwith(df[column], axis=0).values.T)
        product.append(stack)
    return np.abs(np.array(product).swapaxes(1,2))


def remove_edges_via_mean_values(d, cfg, return_means=False):
    """
    From CausalRivers: 
    E.g. as rivers cannot flow from big to small (in most cases), 
    a baseline strategy is to simply remove these impossible 
    links (along with the diagonal) and keep the rest as prediction 
    As an additional restriction, we can only keep one link per node. 
    Here the strategy would be either to choose the biggest or the closest one
    input: pd.Dataframe, reverse_physical: Small to big possible
    outtput: Effect, Cause (nxn)
    """
    if isinstance(d,np.ndarray):
        d = pd.DataFrame(d.T)
    m = np.expand_dims(d.mean().values,axis=1).repeat(len(d.columns),axis=1)
    # compare elements.
    out = m > m.T if cfg.reverse_physical else m < m.T
    #for each column we choose the item which is closest to itself or the biggest available.
    
    if cfg.name == "combo":
        pass
    elif cfg.filter_mode == "min_distance":
        out = min_distance_filter(out, m)
    elif cfg.filter_mode  == "max_distance":
        out = max_distance_filter(out, m, cfg)
    elif (cfg.filter_mode  == "none"):
        pass
    else:
        raise ValueError("Unknown filter mode: {}".format(cfg.filter_mode))
        
    return  (out,None, m) if return_means else (out, None)


def cross_correlation_for_causal_discovery(d, cfg, return_corr=False):
    """ 
    Based on a cross correlation map, this decide which direction an error goes.
    """
    # we actually just have to calculate half here but keep it as it is cleaner.
    if isinstance(d,np.ndarray):
        d = pd.DataFrame(d.T)
    
    corr_map = calc_lagged_cross_corr(d,cfg.max_lag,1)
    out =  corr_map.argmax(axis=2).T > int(corr_map.shape[2] / 2)
    # restrict to the river with the higest cross correlation
    
    
    if cfg.name == "combo":
        pass
    elif cfg.filter_mode == "highest_corr":
        out = corr_filter(out, corr_map)
    elif cfg.filter_mode == "none":
        pass
    else:
        raise ValueError("Unknown filter mode: {}".format(cfg.filter_mode))

    return (out, None, corr_map) if return_corr else (out, None)


def combo_baseline(d, cfg):
    """ 
    This combines both baseline rules.
    """
    cc_out, _, corr_map = cross_correlation_for_causal_discovery(d, cfg, return_corr=True)
    phy_out, _, m = remove_edges_via_mean_values(d, cfg, return_means=True)
    out = cc_out * phy_out
    
    # mask elements that are not relevant anymore. 
    # The same is not necessary for the mean as it is handeled automaticall.
    corr_map[~out] = 0 
    
    if (cfg.filter_mode == "max_distance"):
        out = max_distance_filter(out, m, cfg)
    
    elif cfg.filter_mode == "min_distance":
        out = min_distance_filter(out, m)
        
    elif cfg.filter_mode == "highest_corr":
        out = corr_filter(out, corr_map)
        
    elif cfg.filter_mode == "none":
        pass
    else:
        raise ValueError("Unknown filter mode: {}".format(cfg.filter_mode))
    return out, None


def cross_corr_for_window_causal_graph(d, cfg):
    """
    Uses cross correlation to get link predictions directly.
    """
    
    if isinstance(d,np.ndarray):
        d = pd.DataFrame(d.T)
        
    corr_map = calc_lagged_cross_corr(d,cfg.max_lag)
    # we dont need all of it for consistency we recycle the function.
    corr_map = np.flip(corr_map[:,:,:cfg.max_lag],axis=-1)
    # if we have no variance in the signal this produces nans. catch this here.
    if np.isnan(corr_map).any(): 
        corr_map = np.zeros((d.shape[1],d.shape[1],cfg.max_lag))
    return corr_map, None



