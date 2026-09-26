from common import *

opts = {"display": "off", "random_seed": 1}


# 1. mixed bounded / unbounded variables, as the docstring allows
def mixed():
    b = BADS(
        quad,
        np.array([0.5, 0.5]),
        np.array([0.0, -np.inf]),
        np.array([1.0, np.inf]),
        np.array([0.1, -1.0]),
        np.array([0.9, 1.0]),
        options=dict(opts),
    )
    return b.lower_bounds, b.upper_bounds


trycall("mixed bounded/unbounded", mixed)


# 2. scalar bounds with D = 3
def scal():
    b = BADS(
        quad,
        np.array([0.5, 0.5, 0.5]),
        -1.0,
        1.0,
        -0.5,
        0.9,
        options=dict(opts),
    )
    return b.optim_state["lb_orig"], b.optim_state["plb_orig"]


trycall("scalar bounds D=3", scal)


def scal2():
    b = BADS(quad, np.array([0.5, 0.5, 0.5]), -1.0, 1.0, options=dict(opts))
    return b.optim_state["lb_orig"], b.optim_state["plb_orig"]


trycall("scalar lb/ub only D=3", scal2)


# 3. x0 outside plausible box, inside hard bounds
def outside():
    b = BADS(
        quad,
        np.array([[4.0, 0.0]]),
        np.array([[-10, -10]]),
        np.array([[10, 10]]),
        np.array([[-1, -1]]),
        np.array([[1, 1]]),
        options=dict(opts),
    )
    return (
        b.x0,
        b.optim_state["plb_orig"],
        b.optim_state["pub_orig"],
        b.optim_state["u"],
    )


trycall("x0 outside plausible box", outside)


# 4. log variable, plb = lb, pub = ub
def logvar():
    b = BADS(
        quad,
        np.array([[0.01, 0.0]]),
        np.array([[1e-3, -10]]),
        np.array([[1e3, 10]]),
        None,
        None,
        options=dict(opts),
    )
    return (
        b.x0,
        b.optim_state["plb_orig"],
        b.optim_state["pub_orig"],
        b.var_transf.apply_log_t,
        b.optim_state["lb"],
        b.optim_state["plb"],
    )


trycall("log var, plb=lb", logvar)


def logvar2():
    b = BADS(
        quad,
        np.array([[0.01, 0.0]]),
        np.array([[1e-3, -10]]),
        np.array([[1e3, 10]]),
        np.array([[2e-3, -5]]),
        np.array([[500, 5]]),
        options=dict(opts),
    )
    return (b.x0, b.optim_state["plb_orig"], b.optim_state["pub_orig"])


trycall("log var, plb=2e-3", logvar2)


# 5. x0 with two rows
def tworows():
    b = BADS(
        quad,
        np.array([[0.1, 0.2], [0.3, -0.4]]),
        np.array([[-1, -1]]),
        np.array([[1, 1]]),
        options=dict(opts),
    )
    return b.x0.shape, b.u.shape


trycall("x0 two rows", tworows)


def tworows_opt():
    b = BADS(
        quad,
        np.array([[0.1, 0.2], [0.3, -0.4]]),
        np.array([[-1, -1]]),
        np.array([[1, 1]]),
        options=dict(opts, max_fun_evals=30),
    )
    return b.optimize()["x"]


trycall("x0 two rows optimize", tworows_opt)


def tworows_nolb():
    b = BADS(
        quad,
        np.array([[0.1, 0.2], [0.3, -0.4], [0.2, 0.0]]),
        options=dict(opts),
    )
    return b.optim_state["plb_orig"], b.optim_state["pub_orig"], b.u.shape


trycall("x0 set, no bounds", tworows_nolb)


# 6. fun_values option
def funvals():
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    Y = np.array([[0.05], [0.25]])
    b = BADS(
        quad,
        np.array([0.1, 0.1]),
        np.array([-1, -1]),
        np.array([1, 1]),
        options=dict(opts, fun_values={"X": X, "Y": Y}),
    )
    return b.function_logger.Xn


trycall("fun_values", funvals)


# 7. x0 None with plausible bounds as lists
def lists():
    b = BADS(
        quad, None, [-2, -2], [2, 2], [-1, -1], [1, 1], options=dict(opts)
    )
    return b.x0


trycall("x0 None, bounds lists", lists)


def lists2():
    b = BADS(quad, None, None, None, [-1, -1], [1, 1], options=dict(opts))
    return b.x0


trycall("x0 None, plb lists, no lb", lists2)


def arr1d():
    b = BADS(
        quad,
        None,
        None,
        None,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        options=dict(opts),
    )
    return b.x0


trycall("x0 None, plb 1d arrays, no lb", arr1d)


# 8. partial nan x0
def partial():
    b = BADS(
        quad,
        np.array([0.5, np.nan]),
        np.array([-1, -1]),
        np.array([1, 1]),
        options=dict(opts),
    )
    return b.x0


trycall("x0 partial nan", partial)


def infx0():
    b = BADS(
        quad,
        np.array([0.5, np.inf]),
        np.array([-1, -1]),
        np.array([1, 1]),
        options=dict(opts),
    )
    return b.x0


trycall("x0 inf, bounded", infx0)


def infx0u():
    b = BADS(
        quad,
        np.array([0.5, np.inf]),
        None,
        None,
        np.array([-1, -1]),
        np.array([1, 1]),
        options=dict(opts),
    )
    return b.x0


trycall("x0 inf, unbounded", infx0u)
