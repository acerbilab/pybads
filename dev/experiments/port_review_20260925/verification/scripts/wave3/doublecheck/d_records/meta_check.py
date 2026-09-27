import collections
import glob
import json
import sys

for d in sys.argv[1:]:
    c = collections.Counter()
    starts = []
    ends = []
    crashed = 0
    n = 0
    for p in glob.glob(d + "/*_seed*.json"):
        r = json.load(open(p))
        m = r["meta"]
        n += 1
        key = (
            m["git"]["sha"],
            m["git"]["dirty"],
            m["pybads"],
            m["pybads_source"]["path"],
            m["pybads_source"]["git"]["sha"],
            m["pybads_source"]["git"]["dirty"],
            m["gpyreg"],
            m["gpyreg_source"]["path"],
            m["gpyreg_source"]["git"]["sha"],
            m["python"],
            m["platform"],
            m["numpy"],
            m["scipy"],
            tuple(sorted(m["threads"].items())),
        )
        c[key] += 1
        starts.append(m["started"])
        ends.append(m["finished"])
        crashed += bool(r["final"].get("crashed"))
    print(d.split("/")[-1], n, "runs, crashed", crashed)
    for k, v in c.items():
        print("  ", v, k)
    print("   started", min(starts), "finished", max(ends))
