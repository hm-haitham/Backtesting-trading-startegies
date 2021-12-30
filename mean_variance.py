import pandas as pd
import bahc
import numpy as np

def compute_mean_variance(window=200,filtered=False,step=1):

    preds=pd.Series([0]*window,index=returns.index[0:window])
    for i in range(0,len(returns)-window,step):
        window_returns=returns[i:i+window].copy()
        next_step_returns=returns[i+window:i+window+step].copy()
        if filtered:
            cov=bahc.filterCovariance(np.array(window_returns).T)
        else:
            cov=window_returns.cov()
        inv_cov=np.linalg.inv(cov)
        ones=np.ones(len(inv_cov))
        w_opt=(inv_cov@ones)/(ones.T@inv_cov@ones)

        preds=pd.concat([preds,next_step_returns.apply(lambda x : w_opt@x, axis = 1)])
    return pd.DataFrame(preds)