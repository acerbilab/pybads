"""The first GP of the runs of the ``warmstart`` suite, as the
initialization leaves it, and the logs that two populations gave their
runs.

    python first_gp.py run --root WORKTREE --seeds 0-29 --out FILE.jsonl
    python first_gp.py summary FILE.jsonl...
    python first_gp.py logs REF_DIR NEW_DIR

``run`` builds each run of the suite as ``population.py`` does, with the
``benchmark_targets.py`` and the PyBADS of the checkout ``--root`` (the
earlier run included), calls ``BADS._init_optimization_`` and writes one
JSON line per run: the rows of the start and the initial design
(``_init_rows``), the first GP's training rows, its distinct points and
those among the evaluations given, whether its hyperparameters are MATLAB
BADS's definition values (every log length scale and the log output scale
at 0) and the stages of ``gp_init``. One BLAS thread, as in a population.
``summary`` tabulates the files, one row per file and configuration.
``logs`` checks that the records of two populations name the same log of
evaluations given (``precomputed``) for each (configuration, seed).
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402


def parse_seeds(spec):
    seeds = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def cmd_run(args):
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "dev" / "scripts"))
    import benchmark_targets as bt  # puts ``root`` first on sys.path

    import pybads
    from pybads import BADS
    from pybads.utils.timer.stage_timer import StageTimer

    print(f"[first_gp] pybads {pybads.__file__}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for cfg in bt.suite_configs("warmstart"):
            for seed in parse_seeds(args.seeds):
                prob = cfg.make(seed=seed)
                bargs, options = prob.bads_args()
                bads = BADS(*bargs, options=options, **prob.bads_kwargs())
                bads._stage_timer = StageTimer(lambda: 0.0)
                bads._stage_timer.start()
                gp, _, _, _ = bads._init_optimization_()
                calls = bads._stage_timer.snapshot()["calls"]
                n_given = len(prob.precomputed[1])
                logger = bads.function_logger
                given = logger.X[: logger.Xn + 1][:n_given]
                in_given = [
                    bool(np.any(np.all(given == u, axis=1))) for u in gp.X
                ]
                hyp = gp.get_hyperparameters()[0]
                definition = bool(
                    np.all(hyp["covariance_log_lengthscale"] == 0)
                    and np.all(hyp["covariance_log_outputscale"] == 0)
                )
                row = {
                    "label": cfg.label,
                    "seed": seed,
                    "given_rows": n_given,
                    "init_rows": int(len(bads._init_rows)),
                    "gp_rows": int(gp.X.shape[0]),
                    "gp_distinct": int(np.unique(gp.X, axis=0).shape[0]),
                    "gp_rows_given": int(sum(in_given)),
                    "definition_values": definition,
                    "gp_init_stages": sorted(
                        k for k in calls if k.startswith("gp_init")
                    ),
                }
                f.write(json.dumps(row) + "\n")
                print(f"[first_gp] {json.dumps(row)}", flush=True)
    return 0


def cmd_summary(args):
    print(
        "| file | config | runs | start and design on one point |"
        " first GP on one point | first GP with the definition values |"
        " first GP rows: median [min, max] | of them given: median |"
        " runs with gp_init/gp_rebuild |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for path in args.files:
        rows = [
            json.loads(line) for line in Path(path).read_text().splitlines()
        ]
        for label in sorted({r["label"] for r in rows}):
            rs = [r for r in rows if r["label"] == label]
            n_gp = np.array([r["gp_rows"] for r in rs])
            print(
                f"| `{Path(path).name}` | `{label}` | {len(rs)}"
                f" | {sum(r['init_rows'] == 1 for r in rs)}"
                f" | {sum(r['gp_distinct'] == 1 for r in rs)}"
                f" | {sum(r['definition_values'] for r in rs)}"
                f" | {np.median(n_gp):g} [{n_gp.min()}, {n_gp.max()}]"
                f" | {np.median([r['gp_rows_given'] for r in rs]):g}"
                f" | {sum('gp_init/gp_rebuild' in r['gp_init_stages'] for r in rs)} |"
            )
    return 0


def cmd_logs(args):
    def load(d):
        out = {}
        for p in Path(d).glob("*_seed*.json"):
            r = json.loads(p.read_text())
            out[(r["label"], r["seed"])] = r.get("precomputed")
        return out

    ref, new = load(args.ref), load(args.new)
    keys = sorted(set(ref) & set(new))
    differ = [k for k in keys if ref[k] != new[k] or ref[k] is None]
    print(
        f"[first_gp] {len(keys) - len(differ)} of {len(keys)} pairs name the"
        " same log of evaluations given"
    )
    for k in differ:
        print(f"- differs: {k[0]} seed {k[1]}: {ref[k]} / {new[k]}")
    return 1 if differ else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--seeds", default="0-29")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    lg = sub.add_parser("logs")
    lg.add_argument("ref")
    lg.add_argument("new")
    args = ap.parse_args()
    return {"run": cmd_run, "summary": cmd_summary, "logs": cmd_logs}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
