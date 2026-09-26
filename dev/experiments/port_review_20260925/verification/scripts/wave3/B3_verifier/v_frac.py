"""F7: the ES's fraction of new candidates, port rule vs searchES.m:170-193, same z values."""
import numpy as np
import vhdr  # noqa

rng = np.random.default_rng(0)
lam = 2048
gens = [rng.normal(-0.3 * k, 1.0, size=lam) for k in range(5)]

# MATLAB: us/zold trimmed to lam each iteration; index 1-based; nnew = sum(index(1:ntest) > nold)
zold = np.empty(0)
fr_m = []
for i, znew in enumerate(gens):
    nold = zold.size
    z = np.concatenate([zold, znew])
    index = np.argsort(z, kind="stable") + 1
    N = min(z.size, lam)
    ntest = min(znew.size, nold)
    nnew = np.sum(index[:ntest] > nold)
    fr_m.append(nnew / ntest if ntest else np.nan)
    zold = z[index[:N] - 1]

# Port: z_candidates pooled untrimmed; nold = previous selection size (mu at i = 0)
zc = None
nold_prev = lam
fr_p = []
fr_true = []
for i, znew in enumerate(gens):
    nold = nold_prev
    zc = znew.copy() if i == 0 else np.concatenate([zc, znew])
    N = min(zc.size, lam)
    z_idx = np.argsort(zc)
    ntest = min(znew.size, nold)
    n_new = np.sum(z_idx[0 : ntest + 1] > nold)
    fr_p.append(n_new / ntest)
    first_new = zc.size - znew.size
    fr_true.append(np.sum(z_idx[:ntest] >= first_new) / ntest)
    nold_prev = N
print("iteration (1-based)     :", list(range(1, 6)))
print("MATLAB frac             :", np.round(fr_m, 3).tolist())
print("port frac               :", np.round(fr_p, 3).tolist())
print(
    "true new share, top ntest (port's pool):", np.round(fr_true, 3).tolist()
)
print(
    "frac enters the scale only for 1 < i < Nsearchiter (1-based); default Nsearchiter = 2 -> never"
)
