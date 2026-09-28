"""A noisy run's first incumbent, at the end of its initialization: the raw
minimum of the initial design, the initial GP's prediction there, and the
target's noiseless value there, for each run of a population.

    python w236_first_incumbent.py --seeds 0-89 --out FILE [--only LABELS]
    python w236_first_incumbent.py summary FILE

Each run is built as ``population.py`` builds it (the configuration's start
point, noise stream and options, ``random_seed`` = seed), and ``BADS``
stops after ``_init_optimization_``: the initial design and the fit of the
initial GP. PyBADS and ``benchmark_targets.py`` come from the checkout
that holds this script, as for ``population.py``. Each run appends one
JSON line to FILE; ``summary`` tabulates them per configuration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import benchmark_targets as bt  # noqa: E402  (puts its checkout first)

NOISY = (
    "sphere_D3_homo",
    "ellipsoid_D3_homo",
    "sphere_D3_hetero",
    "ellipsoid_D3_hetero",
    "multisensory_s1_D6_homo",
)


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def first_incumbent(label, seed):
    from pybads import BADS

    prob = bt.find_config(label).make(seed=seed)
    args, options = prob.bads_args()
    bads = BADS(*args, options=options)
    bads.logging_action = []
    gp, _, _, _ = bads._init_optimization_()
    u = np.atleast_2d(bads.u)
    f_mu, f_s2 = gp.predict(u)
    x = bads.var_transf.inverse_transf(u).ravel()
    f_true = prob.f_true(x)
    return {
        "label": label,
        "seed": seed,
        "n_init": int(bads.function_logger.func_count),
        "f_true": f_true,
        "f_min": prob.f_min,
        "noise_sd": prob.noise_sd(x),
        "yval": float(bads.yval),
        # the incumbent's value and SD as the run holds them
        "fval": float(bads.fval),
        "fsd": float(bads.fsd),
        "gp_mu": float(f_mu.item()),
        "gp_sd": float(np.sqrt(f_s2).item()),
    }


def cmd_run(a):
    bt.single_thread_env()
    labels = a.only.split(",") if a.only else list(NOISY)
    with open(a.out, "a", encoding="utf-8") as fh:
        for label in labels:
            for seed in parse_seeds(a.seeds):
                r = first_incumbent(label, seed)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(
                    f"{label} seed {seed}: yval - f = "
                    f"{r['yval'] - r['f_true']:+.3f}, gp - f = "
                    f"{r['gp_mu'] - r['f_true']:+.3f}",
                    flush=True,
                )


def cmd_summary(a):
    rows = [json.loads(s) for s in Path(a.file).read_text().splitlines()]
    labels = [lab for lab in NOISY if any(r["label"] == lab for r in rows)]
    print(
        "| config | runs | initial evaluations | median f - f_min"
        " | median yval - f | median abs(yval - f)"
        " | median GP mean - f | median abs(GP mean - f) | GP closer"
        " | median SD of yval | median GP SD | yval within 2 SD"
        " | GP within 2 SD |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label in labels + ["all"]:
        rs = [r for r in rows if label in ("all", r["label"])]
        f = np.array([r["f_true"] for r in rs])
        f_min = np.array([r["f_min"] for r in rs])
        y = np.array([r["yval"] for r in rs]) - f
        g = np.array([r["gp_mu"] for r in rs]) - f
        s_y = np.array([r["fsd"] for r in rs])
        s_g = np.array([r["gp_sd"] for r in rs])
        n_init = sorted({r["n_init"] for r in rs})
        print(
            f"| {label} | {len(rs)} | {'/'.join(map(str, n_init))}"
            f" | {np.median(f - f_min):.3g}"
            f" | {np.median(y):+.2f} | {np.median(np.abs(y)):.2f}"
            f" | {np.median(g):+.2f} | {np.median(np.abs(g)):.2f}"
            f" | {np.mean(np.abs(g) < np.abs(y)):.2f}"
            f" | {np.median(s_y):.3g} | {np.median(s_g):.3g}"
            f" | {np.mean(np.abs(y) <= 2 * s_y):.2f}"
            f" | {np.mean(np.abs(g) <= 2 * s_g):.2f} |"
        )
    print(
        "\nf is the target's noiseless value at the first incumbent, yval its"
        " observation there, the raw minimum of the initial design, and the"
        " GP mean and SD the initial GP's prediction there. The SD of yval is"
        " the one the run gives it: `noise_size`, 1, or with"
        " `specify_target_noise` the SD that the target returned there, its"
        ' true noise SD. "GP closer": the fraction of runs whose GP mean is'
        " closer to f than yval is."
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        p = argparse.ArgumentParser()
        p.add_argument("cmd")
        p.add_argument("file")
        cmd_summary(p.parse_args())
        return
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only")
    cmd_run(p.parse_args())


if __name__ == "__main__":
    main()
