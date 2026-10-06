"""B7 verifier: (1) random x0 over 20 seeds, D = 3: the design and the start
after it (init only); (2) malformed target outputs (F8 / comparison F7);
(3) finalize and reset_fun_eval_time (F9); (4) add (F10); (5) a noise test
whose second value is NaN or inf (comparison F6a)."""
import warnings

import gpyreg
import numpy as np

import pybads
from pybads import BADS
from pybads.function_logger import FunctionLogger

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)


def quad(x):
    x = np.atleast_2d(x)
    return float(np.sum((x - 0.1) ** 2 * np.arange(1, x.size + 1)))


print("\n(1) random x0, D = 3, 20 seeds", flush=True)
designs, starts, x0s = [], [], []
for seed in range(20):
    D = 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            quad,
            None,
            -5 * np.ones((1, D)),
            5 * np.ones((1, D)),
            -2 * np.ones((1, D)),
            2 * np.ones((1, D)),
            options={"display": "off", "random_seed": seed},
        )
        b.logging_action = []
        b._init_mesh_()
    lg = b.function_logger
    designs.append(lg.X[1 : lg.Xn + 1].copy())
    starts.append(tuple(b.u))
    x0s.append(tuple(lg.X[0]))
same = all(
    np.array_equal(np.sort(designs[0], axis=0), np.sort(d, axis=0))
    for d in designs
)
from collections import Counter  # noqa: E402

cnt = Counter(starts)
print(
    f"distinct x0 {len(set(x0s))}; designs identical in all 20: {same}; "
    f"start after the design: {cnt.most_common(1)[0][1]} of 20 at the same "
    f"design point, {sum(1 for s, x in zip(starts, x0s) if s == x)} stay at "
    f"x0",
    flush=True,
)


def call(level, out, fn_name="v"):
    lg = FunctionLogger(lambda x: out, 2, level == 2, level)
    try:
        v = lg(np.zeros(2))
        return f"accepted -> {v[0]!r}, sd {v[1]!r}"
    except Exception as e:  # noqa: BLE001
        msg = str(e.args[0]).split("\n")[0].strip()[:60]
        extra = (
            " (+FuncError note)"
            if any("FuncError" in str(a) for a in e.args[1:])
            else ""
        )
        return (
            f"{type(e).__name__}: {msg}{extra}; Xn {lg.Xn}, "
            f"X_max_idx {lg.X_max_idx}, func_count {lg.func_count}"
        )


print("\n(2) malformed outputs", flush=True)
for lev, out in [
    (0, np.array([1.0])),
    (0, [1.0]),
    (0, np.array([1.0, 2.0])),
    (0, "a"),
    (0, None),
    (0, 1 + 0j),
    (0, np.complex128(1 + 0j)),
    (0, 1 + 1j),
    (0, True),
    (0, np.nan),
    (0, (1.0, 0.5)),
    (2, (1.0, 0.5)),
    (2, (1.0, np.array([0.5]))),
    (2, (1.0, [0.5])),
    (2, (1.0, None)),
    (2, (1.0, "a")),
    (2, (1.0, [0.5, 0.6])),
    (2, (1.0, np.array([0.5, 0.6]))),
    (2, (1.0, 0.0)),
    (2, (1.0, np.inf)),
    (2, [1.0, 0.5]),
    (2, np.array([1.0, 0.5])),
]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"level {lev}, output {out!r}: {call(lev, out)}", flush=True)

print("\n(3) finalize and reset_fun_eval_time", flush=True)
lg = FunctionLogger(quad, 2, False, 0, cache_size=3)
for i in range(5):
    lg(np.array([0.1 * i, 0.0]))
print(
    f"before: X {lg.X.shape}, n_evals {lg.n_evals.shape}, fun_eval_time "
    f"{lg.fun_eval_time.shape}",
    flush=True,
)
lg.reset_fun_eval_time()
print(
    f"after reset_fun_eval_time: fun_eval_time {lg.fun_eval_time.shape}",
    flush=True,
)
lg.finalize()
print(
    f"after finalize: X {lg.X.shape}, X_flag {lg.X_flag.shape}, n_evals "
    f"{lg.n_evals.shape}",
    flush=True,
)
try:
    lg.n_evals[lg.X_flag]
    print("n_evals[X_flag] works", flush=True)
except IndexError as e:
    print(f"n_evals[X_flag] raises IndexError: {e}", flush=True)
import inspect  # noqa: E402

import pybads.bads.bads as bm  # noqa: E402

src = inspect.getsource(bm)
print(
    "finalize( called in bads.py:",
    "finalize(" in src,
    "; reset_fun_eval_time( called:",
    "reset_fun_eval_time(" in src,
    "; .add( called:",
    ".add(" in src,
    flush=True,
)

print("\n(4) add", flush=True)
for noise_flag, args in [
    (False, (1.0,)),
    (False, (np.array([1.0]),)),
    (True, (1.0,)),
    (False, (1.0, 0.3)),
]:
    lg = FunctionLogger(quad, 2, noise_flag, 2 if noise_flag else 0)
    try:
        v = lg.add(np.zeros(2), *args)
        s = lg.S[0, 0] if noise_flag else None
        print(
            f"noise_flag {noise_flag}, add{args!r}: -> {v}, S[0] {s}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"noise_flag {noise_flag}, add{args!r}: {type(e).__name__}",
            flush=True,
        )

print("\n(5) noise test whose second value is NaN / inf", flush=True)
for second in (np.nan, np.inf):
    calls = {"n": 0}

    def target(x, second=second):
        calls["n"] += 1
        return second if calls["n"] == 2 else quad(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            target,
            0.4 * np.ones((1, 2)),
            -5 * np.ones((1, 2)),
            5 * np.ones((1, 2)),
            -2 * np.ones((1, 2)),
            2 * np.ones((1, 2)),
            options={"display": "off", "random_seed": 0, "max_fun_evals": 50},
        )
        try:
            r = b.optimize()
            print(
                f"second value {second}: run completes, level "
                f"{b.optim_state['uncertainty_handling_level']}",
                flush=True,
            )
        except ValueError as e:
            print(
                f"second value {second}: ValueError at call {calls['n']}: "
                f"{str(e).splitlines()[0].strip()}",
                flush=True,
            )
