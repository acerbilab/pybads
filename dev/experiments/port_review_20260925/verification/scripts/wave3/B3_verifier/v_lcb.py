"""F8 (internal) / F9 (comparison) / B3-K11: acq_fcn_lcb's sqrt_beta and its SD output."""
import numpy as np
import vhdr  # noqa
from capture import capture_states

from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb

st = capture_states(D=3, seed=0, max_fun_evals=40, max_states=1)[0]
gp = st["gp"]
X = np.vstack([st["u"], st["u"] + 0.1])
for val in (
    None,
    2.0,
    2,
    np.float64(2.0),
    np.array(2.0),
    np.array([2.0]),
    np.inf,
    "my_schedule",
    lambda t, d: 2.0,
):
    try:
        z, fmu, fs = acq_fcn_lcb(X, 10, gp, val)
        print(
            f"sqrt_beta={val!r:>28}: ok, z={np.round(np.ravel(z), 4).tolist()}"
        )
    except Exception as e:
        print(f"sqrt_beta={val!r:>28}: {type(e).__name__}: {str(e)[:70]}")
z, fmu, fs = acq_fcn_lcb(X, 10, gp)
_, s2 = gp.predict(X)
print(
    "third output == sqrt(latent variance):",
    np.allclose(fs, np.sqrt(s2)),
    "| == variance:",
    np.allclose(fs, s2),
)
print(
    "docstring lines:",
    [
        l.strip()
        for l in acq_fcn_lcb.__doc__.splitlines()
        if "variance" in l or "f_s" in l
    ],
)
