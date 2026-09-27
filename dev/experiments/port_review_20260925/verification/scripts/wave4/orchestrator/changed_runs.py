"""Per configuration, the runs whose result differs between two populations
(every field of "final" but the wall time), and their median errors."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
changed = defaultdict(list)
err = defaultdict(lambda: ([], []))
for fa in sorted(a.glob("*_seed*.json")):
    fb = b / fa.name
    ra, rb = json.loads(fa.read_text()), json.loads(fb.read_text())
    label = re.sub(r"_seed\d+\.json$", "", fa.name)
    fa_, fb_ = dict(ra["final"]), dict(rb["final"])
    fa_.pop("wall_s", None)
    fb_.pop("wall_s", None)
    err[label][0].append(ra["final"].get("true_error"))
    err[label][1].append(rb["final"].get("true_error"))
    if fa_ != fb_:
        moved = fa_.get("x") != fb_.get("x")
        changed[label].append((fa.name, moved))
total = sum(len(v) for v in changed.values())
print(
    f"{total} runs changed; {sum(m for v in changed.values() for _, m in v)} end at other points"
)
for label in sorted(changed, key=lambda k: -len(changed[k])):
    ea = np.array([e for e in err[label][0] if e is not None], float)
    eb = np.array([e for e in err[label][1] if e is not None], float)
    print(
        f"{label}: {len(changed[label])} changed ({sum(m for _, m in changed[label])} at other points); median error {np.median(ea):.3g} -> {np.median(eb):.3g}"
    )
