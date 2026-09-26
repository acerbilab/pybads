"""F2 (internal) / F4 (comparison): distribution of _get_random_samples_from_priors_."""
import common  # noqa
import numpy as np

from pybads import BADS
from pybads.bads import gaussian_process_train as gpt


def f(x):
    x = np.ravel(x)
    return float(np.sum((x - 0.3) ** 2 * np.array([1.0, 4.0])))


b = BADS(
    f,
    np.array([[1.0, -1.0]]),
    -5 * np.ones((1, 2)),
    5 * np.ones((1, 2)),
    -2 * np.ones((1, 2)),
    2 * np.ones((1, 2)),
    options={"random_seed": 4, "display": "off", "max_fun_evals": 40},
)
b.optimize()
gp = b.iteration_history["gp"][b.optim_state["iter"]]
names = []
for key, val in gp.get_priors().items():
    print(key, val)
rng = np.random.default_rng(7)
S = np.vstack(
    [gpt._get_random_samples_from_priors_(gp, rng) for _ in range(4000)]
)
d = gp.hyperparameters_to_dict(S)
pri = gp.get_priors()
print(
    f"{'name':32s} {'prior mean':>11s} {'prior sd':>9s} {'draw mean':>12s} {'draw sd':>10s}   bounds"
)
bnd = gp.get_bounds()
for key in pri:
    arr = np.array([dd[key] for dd in d])
    if pri[key][0] != "gaussian":
        continue
    mu = np.ravel(pri[key][1][0])
    sd = np.ravel(pri[key][1][1])
    for j in range(arr.shape[1]):
        lo = np.ravel(bnd[key][0])
        hi = np.ravel(bnd[key][1])
        lo = lo[j] if lo.size > 1 else lo[0]
        hi = hi[j] if hi.size > 1 else hi[0]
        frac_out = np.mean((arr[:, j] < lo) | (arr[:, j] > hi))
        print(
            f"{key+'['+str(j)+']':32s} {mu[j if mu.size>1 else 0]:11.3f} {sd[j if sd.size>1 else 0]:9.3f} "
            f"{arr[:, j].mean():12.4g} {arr[:, j].std():10.4g}   [{lo:.2f}, {hi:.2f}] outside: {frac_out:.2f}"
        )
