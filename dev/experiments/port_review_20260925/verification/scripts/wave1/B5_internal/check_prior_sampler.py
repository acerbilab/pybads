import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS


# A GP as BADS builds it, after a short run on a 3-D target
def f(x):
    x = np.atleast_2d(x)
    return float(np.sum((np.array([1.0, 10.0, 0.3]) * x) ** 2)) * 100


b = BADS(
    f,
    np.array([[2.0, 2.0, 2.0]]),
    np.array([[-20.0] * 3]),
    np.array([[20.0] * 3]),
    np.array([[-5.0] * 3]),
    np.array([[5.0] * 3]),
    options={"random_seed": 1, "display": "off", "max_fun_evals": 60},
)
r = b.optimize()
gp = b.iteration_history["gp"][b.optim_state["iter"]]
pri = gp.get_priors()
rng = np.random.default_rng(0)
S = np.vstack(
    [gpt._get_random_samples_from_priors_(gp, rng) for _ in range(4000)]
)
names = []
for k, v in gp.get_hyperparameters()[0].items():
    names += [k] * np.size(v)
print(
    "hyperparameter, prior (type, mean, sd), sampler mean, sampler sd, lower, upper bound"
)
for i, n in enumerate(names):
    p = pri[n]
    j = sum(1 for m in names[:i] if m == n)
    mu = np.ravel(p[1][0])[j] if p[0] == "gaussian" else None
    sd = np.ravel(p[1][1])[j] if p[0] == "gaussian" else None
    print(
        f"{n}[{j}]: prior {p[0]} mu={mu:.3f} sd={sd:.3f} | draws mean={S[:,i].mean():.3f} sd={S[:,i].std():.3f} | bounds [{gp.lower_bounds[i]:.2f}, {gp.upper_bounds[i]:.2f}]"
    )
