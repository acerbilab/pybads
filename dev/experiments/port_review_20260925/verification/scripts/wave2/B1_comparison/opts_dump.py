import os
import re

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads.bads.options import Options, _read_config_file

root = os.path.dirname(pybads.__file__)
bp = root + "/bads/option_configs/basic_bads_options.ini"
ap = root + "/bads/option_configs/advanced_bads_options.ini"
names_b = list(_read_config_file(bp)[:, 0])
names_a = list(_read_config_file(ap)[:, 0])
vals = {}
for D in (1, 2, 6, 20):
    o = Options(bp, evaluation_parameters={"D": D}, user_options=None)
    o.load_options_file(ap, evaluation_parameters={"D": D})
    vals[D] = o
# count reads in package code (excluding testing and ini)
pyfiles = []
for dp, dn, fn in os.walk(root):
    if "testing" in dp or "__pycache__" in dp:
        continue
    for f in fn:
        if f.endswith(".py"):
            pyfiles.append(os.path.join(dp, f))
src = {f: open(f).read() for f in pyfiles}


def reads(name):
    pat = re.compile(r"""["']%s["']""" % re.escape(name))
    hits = []
    for f, s in src.items():
        for i, l in enumerate(s.splitlines(), 1):
            if pat.search(l):
                hits.append(f"{os.path.relpath(f, root)}:{i}")
    return hits


for n in names_b + names_a:
    r = reads(n)
    vs = [vals[D][n] for D in (1, 2, 6, 20)]

    def fmt(v):
        if callable(v):
            return "<lambda>"
        if isinstance(v, np.ndarray):
            return "arr" + str(v.tolist())
        return repr(v)

    print(
        f"{'B' if n in names_b else 'A'} {n:38s} | "
        + " | ".join(fmt(v) for v in vs)
        + f" | reads={len(r)} {r[:3]}"
    )
