"""The film's illustrations of scenes 2 and 3, on the film's landscape: a plain Bayesian optimization (a GP with
a squared-exponential kernel and the lower confidence bound) and a plain coordinate direct search.

Run from the film's folder, dev/film; the surfaces are stored as surface.py describes.

    python -u scripts/illustrations.py [--out illus.js] [--trace trace.js]
"""
import argparse
import json

import landscape as LS
import numpy as np
import surface
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

F, XMIN, _, INFO = LS.make(a=0.08, b=4.0, A=(-0.5, -0.5), tstar=4.0)
NG = 128
g = np.linspace(-5, 5, NG)
GX, GY = np.meshgrid(g, g)  # row = x2, column = x1
GRID = np.column_stack([GX.ravel(), GY.ravel()])
C = np.linspace(-5, 5, 161)
CX, CY = np.meshgrid(C, C)
CAND = np.column_stack(
    [CX.ravel(), CY.ravel()]
)  # where the acquisition is minimized


class GP:
    """Zero-mean GP on standardized values, squared-exponential ARD kernel, a little noise; hyperparameters
    by maximum marginal likelihood, from a few starts."""

    def fit(self, X, y):
        self.X, self.m, self.s = X, y.mean(), y.std() + 1e-12
        z = (y - self.m) / self.s

        def nll(p):
            ell, sf, sn = np.exp(p[:2]), np.exp(p[2]), np.exp(p[3])
            K = self.k(X, X, ell, sf) + (sn**2 + 1e-8) * np.eye(len(X))
            try:
                L = cho_factor(K, lower=True)
            except np.linalg.LinAlgError:
                return 1e10
            a = cho_solve(L, z)
            return 0.5 * z @ a + np.sum(np.log(np.diag(L[0])))

        bounds = [(np.log(0.05), np.log(20))] * 2 + [
            (np.log(0.05), np.log(20)),
            (np.log(1e-3), np.log(0.5)),
        ]
        best = None
        for p0 in (
            [0.0, 0.0, 0.0, -4.0],
            [1.0, 1.0, 0.0, -4.0],
            [-1.0, -1.0, 0.5, -3.0],
        ):
            r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        self.p = best.x
        ell, sf, sn = np.exp(self.p[:2]), np.exp(self.p[2]), np.exp(self.p[3])
        self.ell, self.sf = ell, sf
        K = self.k(X, X, ell, sf) + (sn**2 + 1e-8) * np.eye(len(X))
        self.L = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.L, z)
        return self

    @staticmethod
    def k(A, B, ell, sf):
        d = (A[:, None, :] - B[None, :, :]) / ell
        return sf**2 * np.exp(-0.5 * np.sum(d**2, axis=2))

    def predict(self, Xs):
        Ks = self.k(Xs, self.X, self.ell, self.sf)
        mu = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        var = np.maximum(self.sf**2 - np.sum(Ks * v.T, axis=1), 1e-12)
        return self.m + self.s * mu, self.s * np.sqrt(var)


def bayesopt(X0, n_steps, beta=2.0):
    X = np.array(X0, float)
    y = np.array([F(x) for x in X])
    steps = []
    for k in range(n_steps):
        gp = GP().fit(X, y)
        mu, sd = gp.predict(CAND)
        lcb = mu - beta * sd
        for x in X:  # not a point already evaluated
            lcb[np.all(np.isclose(CAND, x), axis=1)] = np.inf
        i = int(np.argmin(lcb))
        x_new = CAND[i]
        y_new = F(x_new)
        gm, gs = gp.predict(GRID)
        steps.append(
            dict(
                n_before=len(X),
                x=x_new.tolist(),
                y=y_new,
                mu=float(mu[i]),
                sd=float(sd[i]),
                ell=gp.ell.tolist(),
                grid_mu=gm,
                grid_sd=gs,
            )
        )
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
        print(
            f"BO step {k + 1:2d}: x=({x_new[0]:+.3f},{x_new[1]:+.3f}) y={y_new:+9.3f} predicted {mu[i]:+9.3f} +- {sd[i]:.3f}"
            f" best {y.min():+9.3f} ell=({gp.ell[0]:.2f},{gp.ell[1]:.2f})",
            flush=True,
        )
    gp = GP().fit(X, y)
    gm, gs = gp.predict(GRID)
    steps.append(
        dict(
            n_before=len(X),
            x=None,
            y=None,
            mu=None,
            sd=None,
            ell=gp.ell.tolist(),
            grid_mu=gm,
            grid_sd=gs,
        )
    )
    return X, y, steps


def mads_polls(
    f,
    x0,
    step0=2.0,
    step_max=4.0,
    tol=4.0 * 2**-7,
    order=((1, 0), (-1, 0), (0, 1), (0, -1)),
):
    """Plain coordinate direct search, recorded poll by poll: the centre, the step, each arm (evaluated, skipped
    after a success, already known, or outside the box) and whether the poll moved.
    """
    x, fx, step = np.array(x0, float), f(x0), step0
    seen = {tuple(np.round(x, 9)): fx}
    polls, n_evals = [], 1
    while step >= tol and n_evals < 600:
        arms, moved = [], False
        for d in order:
            y = x + step * np.array(d, dtype=float)
            if moved:
                arms.append(dict(x=y.tolist(), status="skipped"))
                continue
            if np.any(y < LS.LB) or np.any(y > LS.UB):
                arms.append(dict(x=y.tolist(), status="outside"))
                continue
            key = tuple(np.round(y, 9))
            if key in seen:
                fy, status = seen[key], "known"
            else:
                fy = f(y)
                seen[key] = fy
                n_evals += 1
                status = "new"
            better = fy < fx - 1e-12
            arms.append(
                dict(
                    x=y.tolist(),
                    f=fy,
                    status=("hit" if better else "miss")
                    + ("" if status == "new" else "-known"),
                    n=n_evals if status == "new" else None,
                )
            )
            if better:
                x_new, f_new, moved = y, fy, True
        polls.append(
            dict(center=x.tolist(), f=fx, step=step, arms=arms, moved=moved)
        )
        if moved:
            x, fx = x_new, f_new
        step = min(2 * step, step_max) if moved else step / 2
    return polls, n_evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--out", default="illus.js")
    ap.add_argument(
        "--trace",
        default="trace.js",
        help="the run, whose initial design the illustration starts from",
    )
    args = ap.parse_args()
    run = json.loads(
        open(args.trace, encoding="utf-8")
        .read()
        .split("=", 1)[1]
        .rstrip()
        .rstrip(";")
    )
    X0 = [e[:2] for e in run["evals"][:6]]
    del X0[1]  # the second evaluation of the start
    X, y, steps = bayesopt(X0, args.steps, beta=args.beta)
    grid, surfaces = surface.encode(
        [s["grid_mu"] for s in steps], [s["grid_sd"] for s in steps], NG
    )

    polls, n_mads = mads_polls(F, LS.X0)
    log = LS.mads(F, LS.X0, step0=2.0)
    assert n_mads == len(log), (n_mads, len(log))
    vals = [v for _, v, _, _ in log]
    print(
        f"MADS: {len(log)} evaluations, within 1 of the minimum after {LS.first_within(vals, -32.715224, 1.0)}",
        flush=True,
    )
    from scipy.optimize import minimize as _min

    tm = min(
        (
            _min(
                F,
                x,
                method="Nelder-Mead",
                options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000),
            )
            for x in ([3.5, -0.5], [3.4, -0.48], [3.6, -0.52])
        ),
        key=lambda r: r.fun,
    )
    out = dict(
        truth=dict(
            xmin=[round(float(v), 6) for v in tm.x], fmin=float(tm.fun)
        ),
        bo=dict(
            points=[
                [round(float(a), 5), round(float(b), 5), round(float(c), 5)]
                for (a, b), c in zip(X, y)
            ],
            n_init=len(X0),
            steps=[
                dict(
                    n_before=s["n_before"],
                    x=s["x"],
                    y=s["y"],
                    mu=s["mu"],
                    sd=s["sd"],
                    ell=s["ell"],
                    mu16=m,
                    sd8=d,
                )
                for s, (m, d) in zip(steps, surfaces)
            ],
            grid=grid,
            beta=args.beta,
        ),
        mads=dict(
            log=[
                [
                    round(float(x[0]), 5),
                    round(float(x[1]), 5),
                    round(float(v), 5),
                    kind,
                    step,
                ]
                for x, v, kind, step in log
            ],
            polls=polls,
            step0=2.0,
            step_max=4.0,
            order=["+x1", "-x1", "+x2", "-x2"],
        ),
    )
    text = (
        "window.BADS_ILLUS = " + json.dumps(out, separators=(",", ":")) + ";\n"
    )
    open(args.out, "w", encoding="utf-8").write(text)
    print(f"wrote {args.out}: {len(text) / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
