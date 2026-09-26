"""B1 verifier, short optimize() runs, one at a time:
 (a) result fields: status, success, exit condition (K3, I-F8, C-F7)
 (b) a bound-method target holding a lock (I-F9, C-F6)
 (c) f_vals (K2, I-F6, C-F9)
 (d) multi-row x0 (I-F7, C-F12)
 (e) overhead accounting in a noisy run (C-F13)
"""
import threading
import time

import common  # noqa: F401
import numpy as np

from pybads import BADS

O = {"display": "off", "random_seed": 3}
sph = lambda x: float(np.sum(np.asarray(x) ** 2))
D = 2
bnd = (np.full(D, -5.0), np.full(D, 5.0), np.full(D, -2.0), np.full(D, 2.0))

print("-- (a) result fields")
for mfe in (30, 200):
    r = BADS(
        sph, np.full(D, 1.3), *bnd, options={**O, "max_fun_evals": mfe}
    ).optimize()
    print(
        f"  max_fun_evals={mfe}: success={r['success']!r} message="
        f"{r['message']!r}"
    )
    print(
        f"    'status' in r: {'status' in r}; fsd={r['fsd']!r} "
        f"({type(r['fsd']).__name__}); keys missing from _keys: "
        f"{sorted(set(r._keys) - set(r.keys()))}"
    )
    try:
        r.status
    except AttributeError as e:
        print("    r.status -> AttributeError", e)

print("\n-- (b) bound-method target holding a lock")


class Model:
    def __init__(self):
        self.lock = threading.Lock()
        self.n = 0

    def nll(self, x):
        with self.lock:
            self.n += 1
        return sph(x)


m = Model()
b = BADS(m.nll, np.full(D, 1.3), *bnd, options={**O, "max_fun_evals": 20})
try:
    r = b.optimize()
    print("  optimize returned; result['fun'] is m.nll:", r["fun"] == m.nll)
except Exception as e:
    print(
        f"  optimize raised {type(e).__name__}: {e} after {m.n} "
        f"evaluations; b.x = {getattr(b, 'x', None)}"
    )


class Callable:
    def __call__(self, x):
        return sph(x)


c = Callable()
r = BADS(
    c, np.full(D, 1.3), *bnd, options={**O, "max_fun_evals": 20}
).optimize()
print(
    "  callable object: result['fun'] is c:",
    r["fun"] is c,
    "; plain function kept by reference:",
    BADS(
        sph, np.full(D, 1.3), *bnd, options={**O, "max_fun_evals": 20}
    ).optimize()["fun"]
    is sph,
)

print("\n-- (c) f_vals")
x0 = np.full(D, 1.3)
b = BADS(
    sph, x0, *bnd, options={**O, "max_fun_evals": 20, "f_vals": [sph(x0)]}
)
print("  cache_active:", b.optim_state["cache_active"])
try:
    b.optimize()
    print("  ran")
except Exception as e:
    print(
        f"  optimize raised {type(e).__name__}: {e}; evaluations "
        f"{b.function_logger.func_count}"
    )

print("\n-- (d) multi-row x0")
b = BADS(
    sph,
    np.array([[0.1, 0.2], [0.3, -0.4]]),
    -np.ones(2),
    np.ones(2),
    options={**O, "max_fun_evals": 20},
)
try:
    b.optimize()
    print("  ran")
except Exception as e:
    print(f"  optimize raised {type(e).__name__}: {e}")

print("\n-- (e) overhead: noisy target sleeping 2 ms per call, 60 evals")
calls = []


def slow(x):
    t = time.perf_counter()
    time.sleep(0.002)
    y = sph(x) + 0.1 * noise.normal()
    calls.append(time.perf_counter() - t)
    return y


noise = np.random.default_rng(11)
b = BADS(
    slow,
    np.full(D, 1.3),
    *bnd,
    options={**O, "max_fun_evals": 60, "uncertainty_handling": True},
)
r = b.optimize()
fl = b.function_logger
print(
    f"  func_count {r['func_count']}, stored rows {fl.Xn + 1}, calls "
    f"{len(calls)}; total_fun_eval_time {fl.total_fun_eval_time:.4f} s, "
    f"sum of call times {np.sum(calls):.4f} s, of the last "
    f"{b.options['noise_final_samples']} calls "
    f"{np.sum(calls[-int(b.options['noise_final_samples']):]):.4f} s"
)
print(
    f"  overhead reported {r['overhead']:.3f}; with every call counted "
    f"{r['total_time'] / np.sum(calls) - 1:.3f}"
)
