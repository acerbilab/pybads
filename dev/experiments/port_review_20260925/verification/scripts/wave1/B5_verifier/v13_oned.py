"""F12 (internal): len_scale at D = 1, and the multi-sample sum."""
import common  # noqa
import numpy as np

from pybads import BADS

b = BADS(
    lambda x: float(
        (np.ravel(x)[0] - 0.4) ** 2 * 3 + np.sin(5 * np.ravel(x)[0])
    ),
    np.array([[1.5]]),
    np.array([[-5.0]]),
    np.array([[5.0]]),
    np.array([[-2.0]]),
    np.array([[2.0]]),
    options={"random_seed": 3, "display": "off", "max_fun_evals": 60},
)
b.optimize()
gp = b.iteration_history["gp"][b.optim_state["iter"]]
print(
    "D=1: fitted lengthscale",
    np.exp(gp.get_hyperparameters()[0]["covariance_log_lengthscale"]),
    "temporary_data len_scale",
    gp.temporary_data["len_scale"],
    "ntrain",
    b.optim_state["ntrain"],
)
# the sum as written, for 1 and 2 samples, against MATLAB's weighted sum with weights 1/N
e = [np.array([2.0, 3.0]), np.array([4.0, 5.0])]
for N in (1, 2):
    ls = np.zeros(2)
    for i in range(N):
        ls += ls + e[i]
    print(
        f"{N} sample(s): python sum {ls}, MATLAB weighted sum {sum(e[i] / N for i in range(N))}"
    )
