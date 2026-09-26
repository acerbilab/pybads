"""I-F3 / C-F1 / K6: p_less as coded (bads.py:2279-2284) against MATLAB's
rule (bads.m:862-869), on the shapes a real gpyreg GP gives through
acq_fcn_lcb."""
import gpyreg as gpr
import numpy as np
import vhdr  # prints paths
from scipy.special import erfc

from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb


def p_less_port(f_pi, D):  # verbatim from bads.py
    f_pi = np.sort(f_pi)[::-1]
    return np.prod(1 - f_pi[0 : np.minimum(D + 1, len(f_pi))])


def p_less_matlab(
    f_pi, D
):  # fpi = sort(fpi,'descend'); prod(1-fpi(1:min(nvars,end)))
    f = np.sort(np.ravel(f_pi))[::-1]
    return np.prod(1 - f[: min(D, len(f))])


# a GP of gpyreg on 25 points in D = 3, as PyBADS builds it
rng = np.random.default_rng(0)
D = 3
X = rng.uniform(-1, 1, (25, D))
y = np.sum(X**2, 1, keepdims=True)
gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.RationalQuadraticARD(),
    mean=gpr.mean_functions.ConstantMean(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
gp.fit(X=X, y=y, options={"n_samples": 0, "opts_N": 1, "init_N": 16}, rng=rng)
u = np.zeros((1, D))
mesh = 0.25
u_poll = np.vstack([u + mesh * np.eye(D), u - mesh * np.eye(D)])
z, f_mu, fs = acq_fcn_lcb(u_poll, 25, gp)
print("shapes: z", z.shape, "f_mu", f_mu.shape, "fs", fs.shape)
f_target = float(np.min(y))
gamma_z = (f_target - 0.0 - f_mu) / fs
f_pi = 0.5 * erfc(-gamma_z / np.sqrt(2))
print("f_pi (poll order):", np.round(np.ravel(f_pi), 4))
print("np.sort on (n,1) changes nothing:", np.array_equal(np.sort(f_pi), f_pi))
print(
    "port p_less =",
    p_less_port(f_pi, D),
    " matlab p_less =",
    p_less_matlab(f_pi, D),
)

# the toy of the reports: the high PoI first in poll order, D = 3, 6 points
toy = np.array([[0.69], [1e-9], [1e-9], [1e-9], [1e-9], [1e-9]])
tol_poi = 1e-6 / D
for name, p in (
    ("port", p_less_port(toy, D)),
    ("matlab", p_less_matlab(toy, D)),
):
    print(
        f"toy {name}: p_less={p:.10f} stop after a good poll: {p > 1 - tol_poi}"
    )
# separate the two effects: sorting only, and D vs D+1 only
fs_ = np.sort(np.ravel(toy))[::-1]
print(
    "sorted, D+1 terms:",
    np.prod(1 - fs_[: D + 1]),
    "| unsorted reversed, D terms:",
    np.prod(1 - np.ravel(toy)[::-1][:D]),
)
# D vs D+1 when fewer than D+1 points remain: 4 points, D = 3
t4 = np.array([[1e-7], [1e-7], [1e-7], [1e-7]])
print(
    "4 left, D=3: port",
    p_less_port(t4, D),
    "matlab",
    p_less_matlab(t4, D),
    "threshold",
    1 - tol_poi,
    "port stops:",
    p_less_port(t4, D) > 1 - tol_poi,
    "matlab stops:",
    p_less_matlab(t4, D) > 1 - tol_poi,
)
