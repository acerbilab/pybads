import numpy as np

exec(open("r3_underflow_runs.py").read().split("well = lambda")[0])


def rosen(z):
    z = np.atleast_1d(z)
    return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2))


def well_rosen(x):
    x = np.atleast_1d(x)
    return float(-1000 * np.exp(-np.sum((x - 3) ** 2) / 8) + rosen(x - 3))


for D in (2, 3):
    for seed in (0, 1):
        run("well+rosen", well_rosen, D, seed)
off = lambda x: float(rosen(np.atleast_1d(x) - 4) + 0.0)
for D in (2, 3):
    run("rosen shifted by 4", off, D, 0)
