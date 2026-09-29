"""Print the calls and times of PyBADS's udist, period_check and
local_gp_fitting in two cProfile dumps written by profile_periodic.py
--cprofile, the run with periodic variables and the one without:

    python cprofile_udist.py ON.prof OFF.prof
"""
import pstats
import sys

print("cProfile of periodic_D3_homo, seed 0, gpyreg b44634f, one BLAS thread:")
print(
    "PyBADS's udist, period_check and local_gp_fitting "
    "(calls, own time s, cumulative s)"
)
for arm, path in zip(("on", "off"), sys.argv[1:3]):
    s = pstats.Stats(path)
    print(f"{arm}: total {s.total_tt:.2f} s")
    for k, v in sorted(s.stats.items()):
        if k[2] in ("udist", "period_check", "local_gp_fitting"):
            print(
                f"  {k[2]:18s} {v[1]:6d} calls  own {v[2]:.3f}  "
                f"cum {v[3]:.3f}  ({1e3 * v[3] / v[1]:.2f} ms per call)"
            )
