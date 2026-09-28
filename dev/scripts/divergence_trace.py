"""Find the first computation at which two seeded runs in one process differ.

Runs the optimization of
`test_run_control.py::test_output_fcn_that_changes_nothing_leaves_the_run_unchanged`
(a 3-D sphere, `random_seed=3`, 80 evaluations) several times in one
process, alternating a run without an output function (even runs) and a run
with the test's output function (odd runs):

- `loop N`: N runs; prints each run's final point and the hash of its
  sequence of evaluations, and how many runs differ from the first.
- `trace N OUT_DIR`: N runs under a profile hook that records a digest of
  the arguments at each call, and of the value at each return, of every
  Python function of PyBADS, gpyreg, `numpy.linalg`, `scipy.linalg` and
  `scipy.optimize`. The first run of each configuration is its reference,
  and each later run is compared with it event by event; at the first event
  that differs, the hook records the local variables of the frame and of
  its nearest traced callers. The earliest such event is then recorded
  again in a run that matches the reference there, and the variables that
  differ between the two runs are printed, frame by frame, in the order of
  their first assignment (both records go to `OUT_DIR/snapshots.pkl`).

An ulp-level difference usually leaves the evaluations unchanged, since the
mesh absorbs it, so the trace finds a computation that the platform does
not repeat in far fewer runs than the test's failure needs. In a frame
where an input agrees and the next variable computed from it does not, the
operation between the two is the one that does not repeat.

Before any run, the clocks of PyBADS's timer and of gpyreg are replaced by
a counter restarted at each run, so that the evaluation times a run records
repeat; two untraced runs fill the caches of the libraries first (SciPy's
LAPACK getters among them), which would otherwise change the calls of the
first traced run.
"""

import hashlib
import os
import pickle
import sys
import time
import types

import gpyreg.gaussian_process
import numpy as np

import pybads.utils.timer.timer
from pybads import BADS

D = 3


class _FakeClock:
    """A clock that advances by 1 ms at each read, restarted at each run,
    so that the times a run records repeat from run to run."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1e-3
        return self.t


_clock = _FakeClock()
pybads.utils.timer.timer.time = types.SimpleNamespace(perf_counter=_clock)
gpyreg.gaussian_process.time = types.SimpleNamespace(time=_clock)


def make_bads(trace, **options):
    def sphere(x):
        trace.append(np.array(x, dtype=float).copy())
        return float(np.sum(np.atleast_2d(x) ** 2))

    opts = {"display": "off", "max_fun_evals": 80, "random_seed": 3}
    opts.update(options)
    return BADS(
        sphere,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    )


def meddle(x, optim_state, state):
    optim_state["iter"] = 1000
    optim_state["mesh_size"] = 0.0
    return False


def run_once(i):
    _clock.t = 0.0
    trace = []
    kw = {"output_fcn": meddle} if i % 2 else {}
    res = make_bads(trace, **kw).optimize()
    return res, np.array(trace)


# ---------------------------------------------------------------- digests

MAX_DEPTH = 2
SKIP_KEYS = {"rng"}


def _feed(obj, h, depth=0):
    if isinstance(obj, np.ndarray):
        h.update(f"A{obj.dtype}{obj.shape}".encode())
        if obj.dtype.hasobject:
            for o in obj.flat:
                _feed(o, h, depth + 1)
        else:
            h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, (bool, int, float, complex, np.generic)):
        h.update(repr(obj).encode())
    elif isinstance(obj, (str, bytes)):
        h.update(obj.encode() if isinstance(obj, str) else obj)
    elif obj is None:
        h.update(b"N")
    elif isinstance(obj, (list, tuple)):
        h.update(f"L{len(obj)}".encode())
        if depth < 4:
            for o in obj:
                _feed(o, h, depth + 1)
    elif isinstance(obj, dict):
        h.update(f"D{len(obj)}".encode())
        if depth < 4:
            for k, v in obj.items():
                if k in SKIP_KEYS or callable(v):
                    continue
                _feed(k, h, depth + 1)
                _feed(v, h, depth + 1)
    elif type(obj).__name__ == "Timer":
        h.update(b"Timer")
    elif depth < MAX_DEPTH and hasattr(obj, "__dict__"):
        h.update(type(obj).__name__.encode())
        _feed(vars(obj), h, depth + 1)
    else:
        h.update(type(obj).__name__.encode())


def digest(obj):
    h = hashlib.blake2b(digest_size=8)
    try:
        _feed(obj, h)
    except Exception as e:  # a digest must never stop the run
        h.update(f"!{type(e).__name__}".encode())
    return h.digest()


# ------------------------------------------------------------------ tracer


def _roots():
    import gpyreg
    import scipy

    import pybads

    np_dir = os.path.dirname(np.__file__)
    sp_dir = os.path.dirname(scipy.__file__)
    return (
        os.path.dirname(pybads.__file__),
        os.path.dirname(gpyreg.__file__),
        os.path.join(np_dir, "linalg"),
        os.path.join(sp_dir, "linalg"),
        os.path.join(sp_dir, "optimize"),
    )


ROOTS = _roots()
_traced = {}


def _is_traced(code):
    t = _traced.get(code)
    if t is None:
        f = code.co_filename
        t = (
            f.startswith(ROOTS)
            and "testing" not in f
            and os.path.join("utils", "timer") not in f
        )
        _traced[code] = t
    return t


def _where(frame):
    code = frame.f_code
    f = code.co_filename
    for r in ROOTS:
        if f.startswith(r):
            f = os.path.basename(r) + f[len(r) :]
            break
    return f"{f}:{code.co_name}:{frame.f_lineno}"


def _stack(frame, n=8):
    out = []
    while frame is not None and len(out) < n:
        if _is_traced(frame.f_code):
            out.append(_where(frame))
        frame = frame.f_back
    return out


def _args_digest(frame):
    code = frame.f_code
    n = code.co_argcount + code.co_kwonlyargcount
    names = code.co_varnames[:n]
    loc = frame.f_locals
    return digest(
        [loc.get(a) for a in names if a not in SKIP_KEYS | {"self", "cls"}]
    )


def _snapshot(frame, levels=4):
    """The digest of each local variable of `frame` and of its nearest
    traced callers, with the values of the numerical ones."""
    out = []
    f = frame
    while f is not None and len(out) < levels:
        if f is frame or _is_traced(f.f_code):
            digests, values = {}, {}
            for name, v in f.f_locals.items():
                if name in SKIP_KEYS or name == "self" or callable(v):
                    continue
                digests[name] = digest(v).hex()
                if isinstance(v, (np.ndarray, np.generic, float, int)):
                    values[name] = np.array(v, copy=True)
            out.append(
                {"where": _where(f), "digests": digests, "values": values}
            )
        f = f.f_back
    return out


class Tracer:
    """Records (event, code, digest) for the traced functions. With `ref`,
    compares each event with the reference on the fly and snapshots the
    frames at the first that differs; with `capture_at`, snapshots the
    frames at that event."""

    def __init__(self, ref=None, capture_at=None):
        self.events = []
        self.ref = ref
        self.mismatch = None
        self.capture_at = capture_at
        self.captured = None

    def __call__(self, frame, event, arg):
        if event not in ("call", "return") or not _is_traced(frame.f_code):
            return
        if self.mismatch is not None:
            return  # nothing more to learn from this run
        d = _args_digest(frame) if event == "call" else digest(arg)
        k = len(self.events)
        rec = (event, frame.f_code, d)
        self.events.append(rec)
        if k == self.capture_at:
            self.captured = {
                "rec": rec,
                "where": _where(frame),
                "frames": _snapshot(frame),
            }
        ref = self.ref
        if ref is not None and (k >= len(ref) or ref[k] != rec):
            r = ref[k] if k < len(ref) else None
            self.mismatch = {
                "k": k,
                "event": event,
                "where": _where(frame),
                "stack": _stack(frame),
                "ref": (r[0], r[1].co_name) if r else None,
                "frames": _snapshot(frame),
            }


def traced_run(i, **kw):
    t = Tracer(**kw)
    sys.setprofile(t)
    try:
        res, T = run_once(i)
    finally:
        sys.setprofile(None)
    return t, res, T


def _func(where):
    return where.rsplit(":", 1)[0]


def compare_snapshots(div, same):
    """Prints, frame by frame, the local variables whose values differ
    between a run that diverged and one that did not, in the order of their
    first assignment."""
    for fd, fs in zip(div, same):
        if _func(fd["where"]) != _func(fs["where"]):
            print(f"  {fd['where']} vs {fs['where']}: different functions")
            break
        dd, ds = fd["digests"], fs["digests"]
        differ = [n for n in dd if dd.get(n) != ds.get(n)]
        line = fs["where"].rsplit(":", 1)[1]
        print(f"  {fd['where']} (the matching run at line {line})")
        print(f"    locals: {list(dd)}")
        print(f"    differ: {differ}")
        for n in differ:
            a, b = fd["values"].get(n), fs["values"].get(n)
            if a is not None and b is not None and a.shape == b.shape:
                with np.errstate(all="ignore"):
                    diff = np.max(np.abs(a - b)) if a.size else 0.0
                print(f"      {n}: shape {a.shape}, max abs diff {diff:.3g}")


# ------------------------------------------------------------------ modes


def mode_loop(n):
    ref = None
    n_diff = 0
    t0 = time.perf_counter()
    for i in range(n):
        res, T = run_once(i)
        h = hashlib.sha256(T.tobytes()).hexdigest()[:12]
        if ref is None:
            ref = T
        same = T.shape == ref.shape and np.array_equal(T, ref)
        n_diff += not same
        print(
            f"run {i:3d} fcn={i % 2} evals={len(T)} hash={h} "
            f"x={res['x']} same={same} t={time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    print(f"LOOP: {n_diff} of {n} runs differ from run 0", flush=True)


def mode_trace(n, out_dir):
    """Runs alternate the two configurations (even: no output function,
    odd: the test's); each run is compared with the first run of its own
    configuration, since the output function adds calls of its own."""
    t0 = time.perf_counter()
    # the first calls fill caches (SciPy's LAPACK getters among them), which
    # changes the calls that later runs make
    run_once(0)
    run_once(1)
    refs = {}
    mismatches = []
    for i in range(n):
        c = i % 2
        if c not in refs:
            t, res, T = traced_run(i)
            refs[c] = (t.events, T)
            print(
                f"reference of fcn={c}: {len(t.events)} events, "
                f"x={res['x']}, t={time.perf_counter() - t0:.0f}s",
                flush=True,
            )
            continue
        ref, T0 = refs[c]
        t, res, T = traced_run(i, ref=ref)
        same_T = T.shape == T0.shape and np.array_equal(T, T0)
        m = t.mismatch
        if m is None and len(t.events) != len(ref):
            m = {"k": len(ref), "event": "end", "where": "", "stack": []}
        print(
            f"run {i:3d} fcn={c} evals_same={same_T} x={res['x']} "
            f"first_diff={None if m is None else m['k']} "
            f"t={time.perf_counter() - t0:.0f}s",
            flush=True,
        )
        if m is not None:
            mismatches.append((m["k"], c, i, m))
            print(f"    {m['event']} at {m['where']} (ref: {m.get('ref')})")
            for s in m["stack"][1:]:
                print(f"      from {s}")
    n_cmp = n - len(refs)
    if not mismatches:
        print(f"TRACE: all {n_cmp} runs repeat their reference event by event")
        return
    print(
        f"TRACE: {len(mismatches)} of {n_cmp} runs differ from their reference"
    )
    k, c, i, m = min(mismatches, key=lambda x: x[:3])
    if "frames" not in m:
        return
    ref = refs[c][0]
    print(f"earliest: run {i}, event {k}; compared with a run that matches")
    for j in range(n):
        t, _, _ = traced_run(c + 2 * j, capture_at=k)
        cap = t.captured
        if cap is not None and cap["rec"] == ref[k]:
            break
    else:
        print("  none of the runs matched the reference at that event")
        return
    compare_snapshots(m["frames"], cap["frames"])
    with open(os.path.join(out_dir, "snapshots.pkl"), "wb") as f:
        ref_cap = {key: v for key, v in cap.items() if key != "rec"}
        pickle.dump({"diverged": m, "reference": ref_cap}, f)


def _at_offset(arr, offset, order="F"):
    """A copy of `arr` whose data start `offset` bytes past a 128-byte
    boundary."""
    buf = np.empty(arr.nbytes + 256, dtype=np.uint8)
    start = (-buf.ctypes.data) % 128 + offset
    out = np.ndarray(
        arr.shape, dtype=arr.dtype, buffer=buf, offset=start, order=order
    )
    out[...] = arr
    return out


def mode_align(n_rep):
    """The linear algebra of a GP fit, on fixed inputs, with the inputs at
    each 8-byte offset from a 128-byte boundary, and repeated with the
    allocator's state shifted before each call (which moves the arrays that
    the call allocates): the number of distinct results, bit for bit."""
    import scipy.linalg
    from gpyreg.gaussian_process import _solve_triangular

    rng = np.random.default_rng(0)
    offsets = range(0, 128, 8)
    print(
        f"{'operation':<24} {'N':>3} {'a offsets':>9} {'b offsets':>9} "
        f"{'repeats':>8}"
    )
    for N in (6, 11, 16, 25, 40, 64, 81):
        X = rng.uniform(-1, 1, size=(N, D))
        d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        K = np.exp(-0.5 * d2 / 0.3**2)
        A = K / 1e-4
        A.flat[:: N + 1] += 1.0
        L = scipy.linalg.cholesky(A, check_finite=False)
        b = rng.normal(size=(N, 1))
        v = rng.normal(size=(N, 1))
        ops = {
            "trtrs (trans=1)": (
                L,
                b,
                lambda a, y: _solve_triangular(a, y, trans=1),
            ),
            "trtrs (trans=0)": (
                L,
                b,
                lambda a, y: _solve_triangular(a, y, trans=0),
            ),
            "potrf": (
                np.asfortranarray(A),
                None,
                lambda a, y: scipy.linalg.cholesky(a, check_finite=False),
            ),
            "matmul NxN @ NxN": (K, A, lambda a, y: a @ y),
            "matmul NxN @ Nx1": (K, v, lambda a, y: a @ y),
            "matmul 1xN @ Nx1": (v.T.copy(), v, lambda a, y: a @ y),
            "np.sum": (K, None, lambda a, y: np.sum(a)),
        }
        for name, (a0, b0, f) in ops.items():
            order = "F" if a0.flags.f_contiguous else "C"
            ra = {
                f(_at_offset(a0, o, order), b0).tobytes()
                if isinstance(f(a0, b0), np.ndarray)
                else repr(f(_at_offset(a0, o, order), b0))
                for o in offsets
            }
            rb = {"-"}
            if b0 is not None:
                ob = "F" if b0.flags.f_contiguous else "C"
                rb = {
                    np.asarray(f(a0, _at_offset(b0, o, ob))).tobytes()
                    for o in offsets
                }
            rr = set()
            junk = []
            for j in range(n_rep):
                junk.append(np.empty(int(rng.integers(1, 64))))
                if len(junk) > 16:
                    junk.pop(int(rng.integers(0, len(junk))))
                rr.add(np.asarray(f(a0, b0)).tobytes())
            nb = "-" if b0 is None else len(rb)
            print(
                f"{name:<24} {N:>3} {len(ra):>9} {nb:>9} {len(rr):>8}",
                flush=True,
            )


def main():
    mode = sys.argv[1]
    n = int(sys.argv[2])
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    os.makedirs(out_dir, exist_ok=True)
    import scipy

    print(
        f"python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}",
        flush=True,
    )
    if mode == "loop":
        mode_loop(n)
    elif mode == "trace":
        mode_trace(n, out_dir)
    elif mode == "align":
        mode_align(n)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
