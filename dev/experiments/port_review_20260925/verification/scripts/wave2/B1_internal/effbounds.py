from common import *

opts = {"display": "off", "random_seed": 0}


def show(name, x0, lb, ub, plb, pub):
    def f():
        b = BADS(
            quad,
            np.array([x0], float),
            np.array([lb], float),
            np.array([ub], float),
            None if plb is None else np.array([plb], float),
            None if pub is None else np.array([pub], float),
            options=dict(opts),
        )
        return dict(
            x0=b.x0.ravel(),
            plb=b.optim_state["plb_orig"].ravel(),
            pub=b.optim_state["pub_orig"].ravel(),
            log=b.var_transf.apply_log_t.ravel(),
            u0=b.u,
        )

    trycall(name, f)


show(
    "rate: lb=0 ub=1000 plb=0.1 pub=10 x0=0.5",
    [0.5, 0.0],
    [0, -1],
    [1000, 1],
    [0.1, -0.5],
    [10, 0.5],
)
show(
    "rate: lb=1e-3 ub=1e3 plb=1e-2 pub=1e2 x0=0.05",
    [0.05, 0.0],
    [1e-3, -1],
    [1e3, 1],
    [1e-2, -0.5],
    [1e2, 0.5],
)
show(
    "asym linear: lb=-1000 ub=1 plb=0 pub=0.99 x0=0.5",
    [0.5, 0.0],
    [-1000, -1],
    [1, 1],
    [0, -0.5],
    [0.99, 0.5],
)
show(
    "asym linear: lb=-100 ub=1 plb=-1 pub=0.99 x0=0.5",
    [0.5, 0.0],
    [-100, -1],
    [1, 1],
    [-1, -0.5],
    [0.99, 0.5],
)
show(
    "log ub=1e10, far plausible",
    [1e8, 0.0],
    [1, -1],
    [1e10, 1],
    [2e7, -0.5],
    [9e9, 0.5],
)
