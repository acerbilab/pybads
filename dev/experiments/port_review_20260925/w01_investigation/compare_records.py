"""Compare the records of a population with those of a reference, run by run:
which fields of `final` differ (wall_s aside), and by how much.

Usage: python compare_records.py REF_DIR NEW_DIR
"""

import glob
import json
import os
import sys

ref_dir, new_dir = sys.argv[1:3]
n = same = 0
for path in sorted(glob.glob(os.path.join(new_dir, "*.json"))):
    new = json.load(open(path))
    ref_path = os.path.join(ref_dir, os.path.basename(path))
    if not os.path.exists(ref_path):
        print("no reference for", os.path.basename(path))
        continue
    ref = json.load(open(ref_path))
    a, b = ref["final"], new["final"]
    diffs = [k for k in a if k != "wall_s" and a[k] != b.get(k)]
    n += 1
    same += not diffs
    notes = []
    for k in diffs:
        if isinstance(a[k], (int, float)) and isinstance(
            b.get(k), (int, float)
        ):
            notes.append(f"{k} {a[k]:.6g}->{b[k]:.6g}")
        else:
            notes.append(k)
    print(
        f"{new['label']:26s} {new['seed']:3d} {new['meta']['git']['sha']}"
        f"{'+' if new['meta']['git']['dirty'] else ''} | "
        + ("same" if not diffs else "; ".join(notes))
    )
print(f"{same} of {n} runs equal in every final field but wall_s")
