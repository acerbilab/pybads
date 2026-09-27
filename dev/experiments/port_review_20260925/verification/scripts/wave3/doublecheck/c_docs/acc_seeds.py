"""1.1.0 with accelerate_mesh_steps 0, -1, -3 over seeds 0-5 and two
targets: the exception and the iteration at which it comes."""
import sys

sys.argv = ["x", "none"]
exec(open("opts_check.py").read().split("which = sys.argv")[0])


def ell(x):
    x = np.ravel(x)
    return float(np.sum(np.array([1.0, 30.0]) * (x - 0.7) ** 2))


for v in (0, -1, -3):
    for name, fun in (("sphere", sphere), ("ellipse", ell)):
        for seed in range(6):
            base = dict(
                display="off",
                random_seed=seed,
                max_fun_evals=100,
                accelerate_mesh_steps=v,
            )
            b = BADS(
                fun,
                np.full(2, 2.0),
                lower_bounds=np.full(2, -5.0),
                upper_bounds=np.full(2, 5.0),
                plausible_lower_bounds=np.full(2, -4.0),
                plausible_upper_bounds=np.full(2, 4.0),
                options=base,
            )
            try:
                b.optimize()
                print(f"v={v} {name} seed={seed}: ran", flush=True)
            except Exception as e:
                print(
                    f"v={v} {name} seed={seed}: {type(e).__name__} at iter {b.optim_state['iter']}",
                    flush=True,
                )
