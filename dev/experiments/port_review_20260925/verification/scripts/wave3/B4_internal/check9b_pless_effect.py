"""As check9, with min_failed_poll_steps = 2 (level 0) and in noisy runs
(level 1 and level 2) at default options."""
import numpy as np

exec(open("check9_pless_effect.py").read().split("def rosen(x):")[0])


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / (D - 1)) * x) ** 2))


class Noisy:
    def __init__(self, seed, sd, spec):
        self.rng = np.random.default_rng(seed)
        self.sd = sd
        self.spec = spec

    def __call__(self, x):
        x = np.ravel(x)
        y = float(np.sum(x**2) + self.sd * self.rng.standard_normal())
        return (y, self.sd) if self.spec else y


D = 4
lb = -5 * np.ones((1, D))
ub = 5 * np.ones((1, D))
plb = -2 * np.ones((1, D))
pub = 2 * np.ones((1, D))
for label, mk, extra in [
    (
        "ellip4 min_failed_poll_steps=2",
        lambda s: ellip,
        {"min_failed_poll_steps": 2},
    ),
    (
        "noisy sphere4 level1",
        lambda s: Noisy(s, 1.0, False),
        {"uncertainty_handling": True},
    ),
    (
        "noisy sphere4 level2",
        lambda s: Noisy(s, 1.0, True),
        {"specify_target_noise": True},
    ),
]:
    for seed in [1, 2]:
        out = []
        for fix in [False, True]:
            MODE["fix"] = fix
            stats.clear()
            o = {"random_seed": seed, "display": "off", "max_fun_evals": 200}
            o.update(extra)
            r = BADS(
                mk(seed), np.full((1, D), 1.5), lb, ub, plb, pub, options=o
            ).optimize()
            out.append(
                "%s: fval %.4g evals %d %s"
                % (
                    "intended" if fix else "coded",
                    r["fval"],
                    r["func_count"],
                    dict(stats),
                )
            )
        print(label, seed, " | ".join(out))
