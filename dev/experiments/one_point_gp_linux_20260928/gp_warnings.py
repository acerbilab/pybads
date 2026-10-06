"""The warnings of the GP layer in seeded runs, by where they arise.

For each run, the ``RuntimeWarning``s and ``UserWarning``s raised inside
``init_and_train_gp`` (the initial GP), inside ``_robust_gp_fit_`` (the
refits of the local GP) and elsewhere, the refits on one distinct point,
and the initial GP's training set, mean and prior of the mean. PyBADS and
``benchmark_targets.py`` come from the checkout that holds this script,
which it puts first on ``sys.path``, as ``dev/scripts/population.py``
does: copy it into a worktree to measure another commit.

    python gp_warnings.py run --configs sphere_band_D3 --seeds 0-29 --out F
    python gp_warnings.py run --band1 --seeds 0-4 --out F
    python gp_warnings.py summary F [F ...]

``--band1`` runs a problem outside the benchmark: at D = 1, the target
``(x - 1)**2 + 10`` with Gaussian noise of SD 1 (a generator seeded by
``1000 + seed``), bounds ``[-5, 5]``, plausible bounds ``[-2, 2]``,
``x0 = 0`` and the non-box constraint ``|x| > 0.005``, which leaves only
``x0`` feasible; ``uncertainty_handling`` is left empty, so that the noise
test finds the noise.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev" / "scripts"))

import benchmark_targets as bt  # noqa: E402

import pybads  # noqa: E402
import pybads.bads.bads as bads_module  # noqa: E402
import pybads.bads.gaussian_process_train as gpt  # noqa: E402
from pybads import BADS  # noqa: E402


def _seeds(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def _kind(w):
    return f"{w.category.__name__}: {str(w.message)[:60]}"


def _band1(seed):
    noise = np.random.default_rng(1000 + seed)

    def fun(x):
        return float((np.ravel(x)[0] - 1.0) ** 2 + 10.0) + float(
            noise.standard_normal()
        )

    args = (
        fun,
        np.zeros(1),
        -5 * np.ones(1),
        5 * np.ones(1),
        -2 * np.ones(1),
        2 * np.ones(1),
    )
    options = {"display": "off", "max_fun_evals": 500, "random_seed": seed}
    return args, dict(non_box_cons=lambda x: np.abs(x[:, 0]) > 0.005), options


def run_one(label, seed):
    if label == "band1":
        args, kwargs, options = _band1(seed)
        bads = BADS(*args, **kwargs, options=options)
        f_min = 10.0
    else:
        prob = bt.find_config(label).make(seed=seed)
        args, options = prob.bads_args()
        bads = BADS(*args, options=options)
        f_min = prob.f_min
    rec = {"label": label, "seed": seed, "init": [], "refit": [], "other": []}
    refits = []
    original_init = bads_module.init_and_train_gp
    original_fit = gpt._robust_gp_fit_

    def init(*a, **k):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = original_init(*a, **k)
        rec["init"] += [_kind(w) for w in caught]
        gp = out[0]
        rec["init_rows"] = int(gp.X.shape[0])
        rec["init_distinct"] = int(np.unique(gp.X, axis=0).shape[0])
        rec["init_y_min"] = float(np.min(gp.y))
        rec["init_mean"] = float(gp.get_hyperparameters()[0]["mean_const"][0])
        _, (mu, sd) = gp.get_priors()["mean_const"]
        rec["init_mean_prior"] = [
            float(np.ravel(mu)[0]),
            float(np.ravel(sd)[0]),
        ]
        return out

    def fit(gp, X, *a, **k):
        refits.append(int(np.unique(X, axis=0).shape[0]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = original_fit(gp, X, *a, **k)
        rec["refit"] += [_kind(w) for w in caught]
        return out

    bads_module.init_and_train_gp = init
    gpt._robust_gp_fit_ = fit
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = bads.optimize()
    finally:
        bads_module.init_and_train_gp = original_init
        gpt._robust_gp_fit_ = original_fit
    rec["other"] = [_kind(w) for w in caught]
    rec["refits"] = len(refits)
    rec["refits_one_point"] = sum(n == 1 for n in refits)
    rec["func_count"] = int(res["func_count"])
    rec["fval"] = float(res["fval"])
    rec["f_min"] = float(f_min)
    rec["x"] = np.ravel(res["x"]).tolist()
    return rec


def cmd_run(ns):
    labels = ["band1"] if ns.band1 else ns.configs.split(",")
    meta = {
        "pybads": str(Path(pybads.__file__).parent),
        "gpyreg": __import__("gpyreg").__file__,
    }
    with open(ns.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for label in labels:
            for seed in _seeds(ns.seeds):
                rec = run_one(label, seed)
                f.write(json.dumps(rec) + "\n")
                print(
                    f"{label} seed {seed}: init {len(rec['init'])} refit "
                    f"{len(rec['refit'])} other {len(rec['other'])} "
                    f"one-point refits {rec['refits_one_point']}",
                    flush=True,
                )


def cmd_summary(ns):
    for path in ns.files:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])["meta"]
        recs = [json.loads(line) for line in lines[1:]]
        print(
            f"## {path}\n\npybads {meta['pybads']}, gpyreg {meta['gpyreg']}\n"
        )
        print(
            "| configuration | runs | runs warned at init | runs warned in "
            "a refit | runs warned elsewhere | runs with a one-point refit "
            "| init on one point | init mean = its target |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for label in dict.fromkeys(r["label"] for r in recs):
            rs = [r for r in recs if r["label"] == label]
            one = [r for r in rs if r["init_distinct"] == 1]
            print(
                f"| `{label}` | {len(rs)} "
                f"| {sum(bool(r['init']) for r in rs)} "
                f"| {sum(bool(r['refit']) for r in rs)} "
                f"| {sum(bool(r['other']) for r in rs)} "
                f"| {sum(r['refits_one_point'] > 0 for r in rs)} "
                f"| {len(one)} "
                f"| {sum(r['init_mean'] == r['init_y_min'] for r in one)} |"
            )
        kinds = {}
        for r in recs:
            for where in ("init", "refit", "other"):
                for k in set(r[where]):
                    kinds[(where, k)] = kinds.get((where, k), 0) + 1
        print("\n| where | warning | runs |\n|---|---|---|")
        for (where, k), n in sorted(kinds.items()):
            print(f"| {where} | {k} | {n} |")
        print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--configs", default="sphere_band_D3")
    r.add_argument("--band1", action="store_true")
    r.add_argument("--seeds", default="0-29")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    ns = p.parse_args(argv)
    {"run": cmd_run, "summary": cmd_summary}[ns.cmd](ns)


if __name__ == "__main__":
    main()
