"""The evaluation at which the first refit of the GP's hyperparameters comes
in the runs of the ``warmstart`` suite.

    python first_refit.py --root CHECKOUT --seeds 0-9 > first_refit.md

Each run is built as ``population.py`` builds it, with the
``benchmark_targets.py`` and the PyBADS of the checkout ``--root`` (the
earlier run included), and stopped at its first call of
``BADS._record_gp_refit_``, the record of a refit; the table gives the
run's ``func_count`` there, per configuration and seed. One BLAS thread, as
in a population.
"""

import argparse
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


class _FirstRefit(Exception):
    pass


def parse_seeds(spec):
    seeds = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def first_refit(bads):
    """The ``func_count`` of the run's first refit, or ``None`` when the run
    ends without one."""
    record = bads._record_gp_refit_

    def stop():
        record()
        raise _FirstRefit(bads.function_logger.func_count)

    bads._record_gp_refit_ = stop
    try:
        bads.optimize()
    except _FirstRefit as e:
        return int(e.args[0])
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", default="0-9")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "dev" / "scripts"))
    import benchmark_targets as bt  # puts ``root`` first on sys.path

    import pybads
    from pybads import BADS

    seeds = parse_seeds(args.seeds)
    print(f"pybads `{pybads.__file__}`, seeds {args.seeds}\n")
    print("| configuration | " + " | ".join(str(s) for s in seeds) + " |")
    print("|---|" + "---|" * len(seeds))
    for cfg in bt.suite_configs("warmstart"):
        counts = []
        for seed in seeds:
            prob = cfg.make(seed=seed)
            bargs, options = prob.bads_args()
            options["display"] = "off"
            bads = BADS(*bargs, options=options, **prob.bads_kwargs())
            counts.append(first_refit(bads))
        cells = ["—" if c is None else str(c) for c in counts]
        print(f"| `{cfg.label}` | " + " | ".join(cells) + " |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
