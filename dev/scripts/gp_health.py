"""Tables of the GP layer's numerical health over a population.

The counters come from ``gp_health_hooks/sitecustomize.py`` (its docstring
says what each one is), one JSON file per run, written by a population run
with that directory first on ``PYTHONPATH`` and ``GP_HEALTH_OUT`` set::

    GP_HEALTH_OUT=<out>/health \\
    PYTHONPATH=dev/scripts/gp_health_hooks:<gpyreg clone> \\
        python -u dev/scripts/population.py run --suite default \\
        --seeds 0-29 --workers 4 --out <out>/pop

Then, from the repository root::

    python dev/scripts/gp_health.py summary <out>/health [...] \\
        [--md FILE] [--csv FILE]

prints one table per topic, one row per configuration, and writes them to
``--md``; ``--csv`` writes one row per run with the counts the tables sum;
``--pop`` names the population directories whose records give each run's
wall time.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load(dirs):
    runs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
            with open(p, encoding="utf-8") as fh:
                runs.append(json.load(fh))
    return runs


def _sum_where(d, pred, field):
    return sum(v[field] for k, v in d.items() if pred(k))


def per_run(h):
    """The counts of one run, flat."""
    r = {"label": h["label"], "seed": h["seed"], "level": h["level"]}
    calls = h["calls"]
    r["fit_calls"] = calls.get("fit", 0)
    r["update_calls"] = calls.get("update", 0)
    r["sethyp_calls"] = calls.get("set_hyperparameters", 0)
    r["predict_calls"] = calls.get("predict", 0)
    ex = h["exceptions"]
    for m in ("fit", "update", "set_hyperparameters", "predict"):
        r[f"{m}_raised"] = sum(v for k, v in ex.items() if k.startswith(m))
    rf = h["robust_fit"]
    r["refits"] = sum(v for k, v in rf.items() if k.startswith("try0:"))
    r["refits_ok_first"] = rf.get("try0:ok", 0)
    ok = sum(v for k, v in rf.items() if k.endswith(":ok"))
    r["refits_ok_later"] = ok - r["refits_ok_first"]
    r["refits_failed"] = r["refits"] - ok
    r["refit_tries_failed"] = sum(
        v for k, v in rf.items() if not k.endswith(":ok")
    )
    r["fit_s_ok"] = h["fit_seconds"]["ok"]
    r["fit_s_raised"] = h["fit_seconds"]["raised"]
    ch = h["chol"]
    for kind in ("nlZ", "post", "lowfactor"):
        sel = {k: v for k, v in ch.items() if k.split("|")[1] == kind}
        r[f"chol_{kind}"] = sum(v["calls"] for v in sel.values())
        r[f"chol_{kind}_inflated"] = sum(v["inflated"] for v in sel.values())
        r[f"chol_{kind}_raised"] = sum(v["raised"] for v in sel.values())
    r["chol_lowrepr"] = sum(
        v["calls"] for k, v in ch.items() if k.endswith("L_chol=False")
    )
    po = h["posteriors"]
    for m in ("fit", "update", "set_hyperparameters"):
        sel = {k: v for k, v in po.items() if k.split("|")[0] == m}
        r[f"post_{m}"] = sum(v["returns"] for v in sel.values())
        r[f"post_{m}_inflated"] = sum(v["inflated"] for v in sel.values())
    r["post_max_mult"] = max([v["max_mult"] for v in po.values()] or [1.0])
    pr = h["predict"]

    def site(k):
        chain = k.split("|")[0]
        if chain.startswith("acq_fcn_lcb<_poll_step_"):
            return "poll_acq"
        if chain.startswith("acq_fcn_lcb"):
            return "search_acq"
        if chain.startswith("_get_target_from_gp_"):
            return "target"
        return "other"

    for s in ("poll_acq", "search_acq", "target", "other"):
        sel = {
            k: v
            for k, v in pr.items()
            if site(k) == s and k.endswith("add_noise=False")
        }
        r[f"{s}_calls"] = sum(v["calls"] for v in sel.values())
        r[f"{s}_calls_zero"] = sum(v["calls_with_zero"] for v in sel.values())
        r[f"{s}_points"] = sum(v["points"] for v in sel.values())
        r[f"{s}_points_zero"] = sum(v["zero_points"] for v in sel.values())
        r[f"{s}_calls_inflated"] = sum(
            v["calls_inflated_post"] for v in sel.values()
        )
    z = h["zero_sd"]
    r["zero_points"] = z["points"]
    r["zero_raw_negative"] = z["raw_negative"]
    r["zero_raw_zero"] = z["raw_zero"]
    r["zero_raw_positive"] = z["raw_positive"]
    r["zero_low_repr"] = z["low_noise_repr"]
    lp = h["log_prior"]
    r["log_prior_calls"] = sum(v["calls"] for v in lp.values())
    r["log_prior_nan"] = sum(v["nan"] for v in lp.values())
    r["log_prior_neginf"] = sum(v["neginf"] for v in lp.values())
    ob = h["obj_fun"]
    r["obj_nan"] = sum(v["nan"] for v in ob.values())
    r["obj_inf"] = sum(v["neginf_or_inf"] for v in ob.values())
    pout = h["priors_outside"]
    for blk in ("cov", "noise", "mean"):
        v = pout.get(blk, {"fits": 0, "outside": 0, "zero_norm": 0})
        r[f"prior_{blk}_outside"] = v["outside"]
        r[f"prior_{blk}_zero_norm"] = v["zero_norm"]
    r["fits_checked"] = pout.get("mean", {}).get("fits", 0)
    t = h["train_n"]
    r["train_min_all"] = t["min_all"]
    r["train_min_distinct"] = t["min_distinct"]
    r["train_small_distinct_1"] = t["small_distinct"].get("1", 0)
    r["train_small_distinct_2"] = t["small_distinct"].get("2", 0)
    r["hook_errors"] = len(h["hook_errors"])
    return r


def _pct(a, b):
    return "—" if b == 0 else f"{100 * a / b:.1f}%"


def _med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def tables(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    labels = sorted(by, key=lambda k: (by[k][0]["level"] or 0, k))
    out = []

    def table(title, header, fn, note=None):
        out.append(f"### {title}\n")
        if note:
            out.append(note + "\n")
        out.append(
            "| " + " | ".join(["configuration", "level"] + header) + " |"
        )
        out.append("|" + " --- |" * (len(header) + 2))
        for lab in labels:
            rs = by[lab]
            lv = sorted({str(r["level"]) for r in rs})
            out.append(
                "| "
                + " | ".join([lab, ",".join(lv)] + [str(c) for c in fn(rs)])
                + " |"
            )
        out.append("")

    S = lambda rs, k: sum(r[k] for r in rs)  # noqa: E731

    table(
        "Refits of the local GP (`_robust_gp_fit_`)",
        [
            "runs",
            "refits",
            "ok at the first try",
            "ok after retries",
            "every try failed",
            "failed tries",
            "fit time in failed tries",
            "failed tries / run time",
        ],
        lambda rs: [
            len(rs),
            S(rs, "refits"),
            _pct(S(rs, "refits_ok_first"), S(rs, "refits")),
            _pct(S(rs, "refits_ok_later"), S(rs, "refits")),
            _pct(S(rs, "refits_failed"), S(rs, "refits")),
            S(rs, "refit_tries_failed"),
            _pct(
                S(rs, "fit_s_raised"),
                S(rs, "fit_s_raised") + S(rs, "fit_s_ok"),
            ),
            _pct(
                sum(r["fit_s_raised"] for r in rs if r.get("wall_s")),
                sum(r["wall_s"] for r in rs if r.get("wall_s")),
            ),
        ],
        "A try is one `gp.fit`; it fails when a factorization of the "
        "objective fails ten times (`LinAlgError`). A refit whose every try "
        "fails keeps the best of its starts (exit flag -1). The run time "
        "is the population record's `wall_s` (`--pop`).",
    )
    table(
        "Factorizations of the training covariance",
        [
            "objective evaluations",
            "inflated",
            "raised",
            "posteriors",
            "inflated",
            "raised",
            "low-noise repr.",
        ],
        lambda rs: [
            S(rs, "chol_nlZ"),
            _pct(S(rs, "chol_nlZ_inflated"), S(rs, "chol_nlZ")),
            _pct(S(rs, "chol_nlZ_raised"), S(rs, "chol_nlZ")),
            S(rs, "chol_post"),
            _pct(S(rs, "chol_post_inflated"), S(rs, "chol_post")),
            _pct(S(rs, "chol_post_raised"), S(rs, "chol_post")),
            _pct(
                S(rs, "chol_lowrepr"),
                S(rs, "chol_nlZ")
                + S(rs, "chol_post")
                + S(rs, "chol_lowfactor"),
            ),
        ],
        "Inflated: the factorization failed at least once and succeeded "
        "with the noise multiplied by ten per failure; raised: it failed "
        "ten times. Low-noise repr.: the share of factorizations with the "
        "noise variance below 1e-6 (gpyreg's `L_chol = False`).",
    )
    table(
        "Posteriors that keep an inflated noise",
        [
            "after fit",
            "after update",
            "after set_hyperparameters",
            "largest multiplier (median over runs)",
            "search predictions on one",
            "poll predictions on one",
        ],
        lambda rs: [
            _pct(S(rs, "post_fit_inflated"), S(rs, "post_fit")),
            _pct(S(rs, "post_update_inflated"), S(rs, "post_update")),
            _pct(
                S(rs, "post_set_hyperparameters_inflated"),
                S(rs, "post_set_hyperparameters"),
            ),
            f"{_med([r['post_max_mult'] for r in rs]):.0e}",
            _pct(
                S(rs, "search_acq_calls_inflated"), S(rs, "search_acq_calls")
            ),
            _pct(S(rs, "poll_acq_calls_inflated"), S(rs, "poll_acq_calls")),
        ],
        "The share of returns of each method whose posterior keeps a noise "
        "multiplier above one, and of the acquisition's calls on such a GP.",
    )
    table(
        "Zero predictive SDs (latent variance returned as exactly 0)",
        [
            "poll acquisitions with one",
            "poll points",
            "search points",
            "target predictions",
            "negative before the clamp",
            "low-noise repr.",
        ],
        lambda rs: [
            _pct(S(rs, "poll_acq_calls_zero"), S(rs, "poll_acq_calls")),
            _pct(S(rs, "poll_acq_points_zero"), S(rs, "poll_acq_points")),
            _pct(S(rs, "search_acq_points_zero"), S(rs, "search_acq_points")),
            _pct(S(rs, "target_points_zero"), S(rs, "target_calls")),
            _pct(S(rs, "zero_raw_negative"), S(rs, "zero_points")),
            _pct(S(rs, "zero_low_repr"), S(rs, "zero_points")),
        ],
        "A poll acquisition with a zero SD makes the poll's `gamma_z` "
        "infinite and marks the GP unreliable (W3-28).",
    )
    table(
        "Log priors, priors outside their bounds, small training sets",
        [
            "NaN log priors",
            "-inf log priors",
            "NaN objectives",
            "fits with the mean's prior outside its bounds",
            "... the covariance's",
            "... the noise's",
            "runs with 2 or fewer distinct training points",
            "hook errors",
        ],
        lambda rs: [
            S(rs, "log_prior_nan"),
            S(rs, "log_prior_neginf"),
            S(rs, "obj_nan"),
            _pct(S(rs, "prior_mean_outside"), S(rs, "fits_checked")),
            _pct(S(rs, "prior_cov_outside"), S(rs, "fits_checked")),
            _pct(S(rs, "prior_noise_outside"), S(rs, "fits_checked")),
            sum(
                1
                for r in rs
                if r["train_min_distinct"] is not None
                and r["train_min_distinct"] <= 2
            ),
            S(rs, "hook_errors"),
        ],
    )
    return "\n".join(out)


def histograms(runs):
    """The zero-SD histograms, summed over the runs, by level."""
    out = ["### Zero SDs: the value before the clamp and where they occur\n"]
    for lv in sorted({h["level"] for h in runs}, key=lambda x: (x is None, x)):
        hs = [h for h in runs if h["level"] == lv]
        agg = {
            k: defaultdict(int)
            for k in ("raw_rel_log10_hist", "dist_hist", "snr_log10_hist")
        }
        for h in hs:
            for k in agg:
                for b, n in h["zero_sd"][k].items():
                    agg[k][b] += n
        n = sum(h["zero_sd"]["points"] for h in hs)
        if n == 0:
            continue
        out.append(f"Level {lv}, {n} points.\n")
        for k, title in (
            ("raw_rel_log10_hist", "floor(log10(|raw| / kss))"),
            ("dist_hist", "distance to the nearest training input (ell)"),
            ("snr_log10_hist", "floor(log10(kss / effective noise))"),
        ):
            items = sorted(
                agg[k].items(),
                key=lambda kv: (
                    float(kv[0].lstrip("<>="))
                    if kv[0] not in ("-inf",)
                    else -1e9
                ),
            )
            cells = ", ".join(f"{b}: {v}" for b, v in items)
            out.append(f"- {title}: {cells}")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--md")
    s.add_argument("--csv")
    s.add_argument(
        "--pop",
        nargs="*",
        default=[],
        help="population directories whose records give each run's wall_s",
    )
    a = ap.parse_args()
    runs = load(a.dirs)
    rows = [per_run(h) for h in runs]
    wall = {}
    for d in a.pop:
        for p in glob.glob(os.path.join(d, "*_seed*.json")):
            with open(p, encoding="utf-8") as fh:
                rec = json.load(fh)
            wall[(rec["label"], rec["seed"])] = rec["final"]["wall_s"]
    for r in rows:
        r["wall_s"] = wall.get((r["label"], r["seed"]))
    text = tables(rows) + "\n" + histograms(runs)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(text.rstrip("\n") + "\n")
    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh, fieldnames=list(rows[0]), lineterminator="\n"
            )
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
