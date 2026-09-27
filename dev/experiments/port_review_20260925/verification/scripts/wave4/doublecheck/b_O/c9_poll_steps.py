"""W4-20: the poll's steps in the normalized coordinates u and in the
original coordinates x, at default options, for a linear map and for a
variable that the default nonlinear_scaling maps through a log."""
import warnings

import hdr  # noqa
import numpy as np

from pybads import BADS

warnings.simplefilter("ignore")


def steps(lb, ub, plb, pub, x0, f):
    rec = []
    orig = BADS._poll_step_

    def poll(self, gp):
        n0 = self.function_logger.Xn
        u0 = np.ravel(self.u).copy()
        out = orig(self, gp)
        fl = self.function_logger
        U = fl.X[n0 + 1 : fl.Xn + 1]
        X = fl.X_orig[n0 + 1 : fl.Xn + 1]
        x0_ = np.ravel(self.var_transf.inverse_transf(u0))
        rec.append((u0, x0_, U - u0, X - x0_, self.optim_state["mesh_size"]))
        return out

    BADS._poll_step_ = poll
    try:
        b = BADS(
            f,
            x0,
            lb,
            ub,
            plb,
            pub,
            options={"display": "off", "random_seed": 1, "max_fun_evals": 60},
        )
        b.optimize()
    finally:
        BADS._poll_step_ = orig
    print("  log-transformed:", np.ravel(b.var_transf.apply_log_t))
    for u0, x0_, dU, dX, ms in rec[:3]:
        print("  u steps:", np.round(dU, 6).tolist())
        print(
            "  x steps:",
            np.round(dX, 6).tolist(),
            " x at incumbent",
            np.round(x0_, 4).tolist(),
        )


print("linear map, plausible widths 2 and 20:")
steps(
    np.array([[-10, -100]]),
    np.array([[10, 100]]),
    np.array([[-1, -10]]),
    np.array([[1, 10]]),
    np.array([[0.3, 3.0]]),
    lambda x: float(
        (np.ravel(x)[0] - 0.2) ** 2 + ((np.ravel(x)[1] - 2) / 10) ** 2
    ),
)
print("positive bounds, pub/plb = 100 on x2 (log map at default):")
steps(
    np.array([[-10, 0.01]]),
    np.array([[10, 1000]]),
    np.array([[-1, 0.1]]),
    np.array([[1, 10]]),
    np.array([[0.3, 3.0]]),
    lambda x: float(
        (np.ravel(x)[0] - 0.2) ** 2 + np.log(np.ravel(x)[1] / 2) ** 2
    ),
)
