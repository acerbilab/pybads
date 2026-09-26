import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.bads as bb
from pybads import BADS

orig_init = bb.BADS._init_optim_state_


def swapped_back(self):
    os_ = orig_init(self)
    os_["plb"], os_["pub"] = (
        os_["pub"].copy(),
        os_["plb"].copy(),
    )  # plb = lower, pub = upper
    return os_


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


D = 4
res = {}
for variant in ["as is", "plb/pub named correctly"]:
    bb.BADS._init_optim_state_ = (
        orig_init if variant == "as is" else swapped_back
    )
    vals = []
    for seed in range(5):
        b = BADS(
            ell,
            np.full((1, D), 1.5),
            np.full((1, D), -np.inf),
            np.full((1, D), np.inf),
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 150,
            },
        )
        r = b.optimize()
        ps = b.iteration_history["gp"][b.optim_state["iter"]].temporary_data[
            "poll_scale"
        ]
        vals.append(r["fval"])
        print(
            variant,
            seed,
            f"fval {r['fval']:.3g}",
            "final poll_scale",
            np.round(ps, 3),
        )
    res[variant] = vals
print({k: np.round(np.log10(v), 2).tolist() for k, v in res.items()})
