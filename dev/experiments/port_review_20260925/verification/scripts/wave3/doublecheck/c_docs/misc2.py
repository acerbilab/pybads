"""acq_hedge=True, skip_poll=False, search_optimize=True, poll_method and
poll_acq_fcn changed; improvement_quantile as a string."""
import sys

sys.argv = ["x", "none"]
exec(open("opts_check.py").read().split("which = sys.argv")[0])
r0 = run("default", {})
run("acq_hedge=True", {"acq_hedge": True})
r1 = run("skip_poll=False", {"skip_poll": False})
r2 = run("search_optimize=True", {"search_optimize": True})
r3 = run("poll_method='nonexistent'", {"poll_method": "nonexistent"})
r4 = run("poll_acq_fcn=('acq_LCB', 50.0)", {"poll_acq_fcn": ("acq_LCB", 50.0)})
r5 = run("search_improve_frac=0.5", {"search_improve_frac": 0.5})
for lab, r in (
    ("skip_poll", r1),
    ("search_optimize", r2),
    ("poll_method", r3),
    ("poll_acq_fcn", r4),
    ("search_improve_frac", r5),
):
    if r is not None:
        print(
            f"{lab}: identical to default: {r['fval'] == r0['fval'] and r['func_count'] == r0['func_count']}",
            flush=True,
        )
run("improvement_quantile='0.3'", {"improvement_quantile": "0.3"})
