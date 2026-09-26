import sys

sys.path.insert(0, "/home/user/pybads-fix-C/pybads/testing/bads")
import numpy as np
from test_run_control import _make_bads, _Recorder

cases = {
    "default200": dict(max_fun_evals=200),
    "mfe30": dict(max_fun_evals=30),
    "maxiter2": dict(max_iter=2, max_fun_evals=200),
    "mfe1": dict(max_fun_evals=1),
    "tolmesh1e-2": dict(tol_mesh=1e-2, max_fun_evals=200),
    "tolmesh1e-3": dict(tol_mesh=1e-3, max_fun_evals=200),
    "tolfun1e-12": dict(tol_fun=1e-12, max_fun_evals=200),
    "outfcn": dict(output_fcn=_Recorder(stop_at=lambda n: n == 3)),
}
for k, o in cases.items():
    b = _make_bads(**o)
    r = b.optimize()
    print(
        k,
        r["func_count"],
        r["iterations"],
        r.get("status"),
        type(r["fsd"]).__name__,
        repr(r["fsd"]),
        r["message"],
    )
