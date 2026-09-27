#!/bin/bash
S=/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad
while read lab seed; do echo "fd8641d $lab $seed"; echo "1f7c8ee $lab $seed"; done < $S/bisect/moved.txt | xargs -P 4 -L 1 $S/orch/one_run.sh > $S/bisect/attrib_runs.txt 2>&1
python3 - <<'PY'
import json, os
S="/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad"
ref="/home/user/pybads/dev/experiments/population_linux_wave2_20260926"; g0="/home/user/pybads/dev/scripts/runs/population/g0_batch1_a1bf658"
def fin(p):
    d=json.load(open(p))["final"]; d.pop("wall_s",None); return d
same_parent=same_w314=0; bad=[]
for line in open(S+"/bisect/moved.txt"):
    lab,seed=line.split(); n=f"{lab}_seed{seed}.json"
    a=fin(f"{S}/bisect/fd8641d_{lab}_{seed}/{n}")==fin(f"{ref}/{n}")
    b=fin(f"{S}/bisect/1f7c8ee_{lab}_{seed}/{n}")==fin(f"{g0}/{n}")
    same_parent+=a; same_w314+=b
    if not (a and b): bad.append((lab,seed,a,b))
print(f"parent of W3-14 equal to the reference: {same_parent}/31; W3-14 equal to batch 1: {same_w314}/31")
print("not explained:", bad)
PY
