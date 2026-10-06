"""Count, per configuration, the runs of NEW whose result equals REF's
exactly (x0, x, fval, fsd, func_count, iterations, message)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ref, new = Path(sys.argv[1]), Path(sys.argv[2])
KEYS = ("x", "fval", "fsd", "func_count", "iterations", "message")
same, total, diff = defaultdict(int), defaultdict(int), defaultdict(list)
missing = []
for p in sorted(new.glob("*_seed*.json")):
    q = ref / p.name
    if not q.exists():
        missing.append(p.name)
        continue
    a, b = json.loads(q.read_text()), json.loads(p.read_text())
    lab = b["label"]
    total[lab] += 1
    eq = a["x0"] == b["x0"] and all(
        a["final"][k] == b["final"][k] for k in KEYS
    )
    if eq:
        same[lab] += 1
    else:
        diff[lab].append(b["seed"])
for lab in total:
    d = sorted(diff[lab])
    print(
        f"{lab:26s} identical {same[lab]:3d}/{total[lab]:3d}"
        + (f"  differ at seeds {d}" if d else "")
    )
print(
    f"all: {sum(same.values())}/{sum(total.values())} identical;"
    f" {len(missing)} records without a counterpart in REF"
    + (
        f" ({sorted({m.rsplit('_seed', 1)[0] for m in missing})})"
        if missing
        else ""
    )
)
