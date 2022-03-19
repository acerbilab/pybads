
import numpy as np 


def es_update(mu, lamb):

    # Create es vector
    es = dict()
    es['mu'] = mu.copy()
    es['lambda'] = lamb.copy()

    tot = mu + lamb
    sqrt_tot = np.sqrt(np.arange(1, tot+1))
    w = np.ceil((1. / sqrt_tot) / np.sum((1. / sqrt_tot) * lamb))
    nonzero = np.sum(w > 0)
    while (np.sum(w) - lamb) > nonzero:
        w = np.max(0, w-1)
        nonzero = np.sum(w > 0)
    delta = sum(w) - lamb
    lastnonzero = np.argwhere(w > 0)[-1]
    w[np.max(0, lastnonzero - delta):lastnonzero + 1] = w[np.max(0, lastnonzero - delta): lastnonzero + 1] - 1 

    # Create selection mask
    cw = np.cumsum(w) - w + 1
    idx = np.zeros(np.max(cw))
    idx[cw] = 1
    es['select_max'] = np.cumsum(idx[0:-1])
    return es

