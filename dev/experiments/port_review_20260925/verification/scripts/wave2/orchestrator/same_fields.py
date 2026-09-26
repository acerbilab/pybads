"""same_fields.py REF NEW: which leaf fields differ between two populations, run by run (timings left out)."""
import collections
import glob
import json
import os
import sys

ref, new = sys.argv[1:3]


def load(d):
    return {
        os.path.basename(p): json.load(open(p))
        for p in glob.glob(os.path.join(d, "*_seed*.json"))
    }


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    else:
        yield path, x


a, b = load(ref), load(new)
count = collections.Counter()
example = {}
for k in a:
    la, lb = dict(leaves(a[k])), dict(leaves(b[k]))
    for f in sorted(set(la) | set(lb)):
        if f.startswith("meta") or "time" in f or "overhead" in f:
            continue
        if la.get(f, "<absent>") != lb.get(f, "<absent>"):
            count[f] += 1
            example.setdefault(
                f, (k, la.get(f, "<absent>"), lb.get(f, "<absent>"))
            )
for f, n in count.most_common():
    k, va, vb = example[f]
    print(f"{f}: {n} runs; e.g. {k}: {str(va)[:60]!r} -> {str(vb)[:60]!r}")
