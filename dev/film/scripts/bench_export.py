"""The comparison of the closing scene: the six panels of Figs 2 and 3 of Acerbi & Ma (2017), recomputed from the
benchmark's result files as the benchmark's own 2017 code drew them: each optimizer's
per-dataset curve of fraction solved, interpolated on 201 log-spaced points and averaged over the study's datasets.
Deterministic studies plot it against function evaluations / D (10 to 500, method 'FS'); noisy ones against the error
tolerance at the end of the budget (10 down to 0.1, method 'FST'). Writes bench.js for film.html, and with --check a
plot of the six panels to compare with the published figures. DIR holds the benchmark's result files, one folder
per study, which are not public.

    python -u scripts/bench_export.py --data DIR [--out bench.js] [--check OUT/bench.png]
"""
import argparse
import json

import numpy as np
import scipy.io as sio

STUDIES = [  # folder, title on screen, noisy, parameters
    ("ccn17-visvest", "causal inference", False, "10"),
    ("ccn17-adler2016", "Bayesian confidence", False, "13"),
    ("ccn17-goris2014", "neuronal selectivity", False, "12"),
    ("ccn17-vandenberg2017", "word recognition memory", True, "6 and 9"),
    ("ccn17-targetloc", "target detection", True, "6"),
    ("ccn17-vanopheusden2016", "combinatorial game playing", True, "10"),
]
NX = 201


def leaves(node, path=()):
    """The curves of a cache: (path of field names, leaf) for every struct that holds 'yy'."""
    names = node._fieldnames
    if "yy" in names:
        yield path, node
        return
    for n in names:
        v = getattr(node, n)
        if hasattr(v, "_fieldnames"):
            yield from leaves(v, path + (n,))


def study_curves(path, noisy):
    bd = sio.loadmat(
        path,
        variable_names=["benchdata"],
        squeeze_me=True,
        struct_as_record=False,
    )["benchdata"]
    sums, counts, xrange = {}, {}, None
    for (f1, f2, f3), leaf in (
        (p, l) for p, l in leaves(bd) if p[0] != "options"
    ):
        xx, yy = np.atleast_1d(leaf.xx).astype(float), np.atleast_1d(
            leaf.yy
        ).astype(float)
        if noisy:  # 'FST': from the largest tolerance to the smallest
            xr = np.exp(np.linspace(np.log(xx.max()), np.log(xx.min()), NX))
        else:  # 'FS': 10 to 500 evaluations per parameter
            xr = np.exp(np.linspace(np.log(10), np.log(500), NX))
        order = np.argsort(xx)
        y = np.interp(xr, xx[order], yy[order], left=np.nan, right=np.nan)
        xrange = xr if xrange is None else xrange
        assert np.allclose(xrange, xr), (path, f3)
        sums[f3] = sums.get(f3, 0) + y
        counts[f3] = counts.get(f3, 0) + 1
    return xrange, {k[3:]: sums[k] / counts[k] for k in sums}, counts


def kind(algo):
    if algo.startswith("bads"):
        return "bads_overhead" if algo.endswith("overhead") else "bads"
    return "bo" if algo.startswith("bayesopt") else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        required=True,
        help="the folder of the benchmark's result files",
    )
    ap.add_argument("--out", default="bench.js")
    ap.add_argument("--check", default=None)
    args = ap.parse_args()
    studies = []
    for folder, title, noisy, dims in STUDIES:
        x, curves, counts = study_curves(
            f"{args.data}/{folder}/benchdata_1.mat", noisy
        )
        ranked = sorted(curves, key=lambda a: -np.nanmean(curves[a]))
        print(
            f"{title} ({'noisy' if noisy else 'deterministic'}, D = {dims}): {len(curves)} curves, datasets per curve {sorted(set(counts.values()))}",
            flush=True,
        )
        for a in ranked:
            print(f"   {np.nanmean(curves[a]):.3f}  {a}", flush=True)
        studies.append(
            dict(
                folder=folder,
                title=title,
                noisy=noisy,
                dims=dims,
                x=[round(float(v), 4) for v in x],
                algos=[
                    dict(
                        id=a,
                        kind=kind(a),
                        mean=round(float(np.nanmean(curves[a])), 4),
                        y=[
                            None if np.isnan(v) else round(float(v), 4)
                            for v in curves[a]
                        ],
                    )
                    for a in ranked
                ],
            )
        )
    out = dict(
        source="Acerbi & Ma (2017), NeurIPS, Figs 2 and 3; caches benchdata_1.mat of the six CCN17 studies",
        studies=studies,
    )
    text = (
        "window.BADS_BENCH = " + json.dumps(out, separators=(",", ":")) + ";\n"
    )
    open(args.out, "w", encoding="utf-8").write(text)
    print(f"wrote {args.out}: {len(text) / 1e3:.0f} kB", flush=True)
    if args.check:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3, figsize=(18, 9))
        for ax, s in zip(axs.ravel(), studies):
            for a in s["algos"]:
                y = np.array([np.nan if v is None else v for v in a["y"]])
                st = dict(
                    bads=("k", "--", 2.5),
                    bads_overhead=("0.5", "--", 2),
                    bo=("m", "-", 1.5),
                ).get(a["kind"], ("0.7", "-", 1))
                ax.plot(
                    s["x"], y, color=st[0], ls=st[1], lw=st[2], label=a["id"]
                )
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_title(s["title"])
            ax.grid(True, alpha=0.3)
            if s["noisy"]:
                ax.invert_xaxis()
            ax.legend(
                fontsize=6, loc="upper right" if s["noisy"] else "upper left"
            )
        fig.tight_layout()
        fig.savefig(args.check, dpi=80)
        print("wrote", args.check, flush=True)


if __name__ == "__main__":
    main()
