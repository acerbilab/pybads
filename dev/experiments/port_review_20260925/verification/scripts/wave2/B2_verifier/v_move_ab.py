"""I-F1/C-F4: effect of also moving u_best at the re-estimation's move (a
variant; not MATLAB's behaviour). Noisy sphere D=2, 150 evaluations."""
import common
import numpy as np

from pybads import BADS


class Variant(BADS):
    @property
    def best_u(self):
        return self.u_best

    @best_u.setter
    def best_u(self, v):
        self.u_best = np.array(v).copy()


res = []
for seed in range(10):
    out = []
    for cls in (BADS, Variant):
        nrng = np.random.default_rng(200 + seed)
        f = lambda x: float(
            np.sum(np.ravel(x) ** 2) + 0.5 * nrng.standard_normal()
        )
        b = cls(
            f,
            np.array([1.5, -1.0]),
            np.full(2, -5.0),
            np.full(2, 5.0),
            np.full(2, -2.0),
            np.full(2, 2.0),
            options=dict(
                uncertainty_handling=True,
                max_fun_evals=150,
                random_seed=seed,
                display="off",
            ),
        )
        r = b.optimize()
        out.append(
            (
                float(np.sum(np.ravel(r["x"]) ** 2)),
                r["func_count"],
                r["iterations"],
            )
        )
    res.append(out)
    print(
        f"seed {seed}: port err {out[0][0]:.4f} ({out[0][1]} evals) | variant err {out[1][0]:.4f} ({out[1][1]} evals)"
    )
e = np.array([[o[0][0], o[1][0]] for o in res])
print(
    "changed runs",
    int(np.sum(~np.isclose(e[:, 0], e[:, 1]))),
    "of",
    len(e),
    "| median err port %.4f variant %.4f | variant better in %d"
    % (
        np.median(e[:, 0]),
        np.median(e[:, 1]),
        int(np.sum(e[:, 1] < e[:, 0] - 1e-12)),
    ),
)
