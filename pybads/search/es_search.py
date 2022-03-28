
import numpy as np 


def es_update(mu, lamb):

    # Create es vector
    es = dict()
    es['mu'] = mu
    es['lambda'] = lamb

    tot = mu + lamb
    sqrt_tot = np.sqrt(np.arange(1, tot+1))
    w = np.ceil((1. / sqrt_tot) / np.sum((1. / sqrt_tot)) * lamb).astype(int)
    nonzero = np.sum(w > 0)
    while (np.sum(w) - lamb) > nonzero:
        w = np.maximum(0, w-1)
        nonzero = np.sum(w > 0)
    delta = np.sum(w) - lamb
    lastnonzero = (np.argwhere(w > 0)[-1]).item()
    strt_point = np.maximum(0, lastnonzero - int(delta.item()) +1 ).item()
    w[strt_point : lastnonzero] = w[strt_point: lastnonzero] - 1 

    # Create selection mask
    cw = np.cumsum(w) - w 
    idx = np.zeros(np.max(cw)+1, dtype=int)
    idx[cw] = 1
    es['select_max'] = np.cumsum(idx[0:-1])
    return es



