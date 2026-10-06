#!/bin/bash
# usage: pick.sh COMMIT [cl-op args...]   (cl-op: "fixed FILE", "changed FILE", "upgrading FILE", "extend TITLE FILE"; several separated by ';;')
SP=/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad
cd /home/user/pybads
c=$1; shift
git cherry-pick $c > $SP/orch/pick_out.txt 2>&1 || { echo "CHERRY-PICK FAILED for $c"; cat $SP/orch/pick_out.txt | tail -5; exit 1; }
ops="$*"
if [ -n "$ops" ]; then
  IFS=';;' read -ra parts <<< "$ops"
  python3 - "$ops" <<'PY'
import sys, subprocess, shlex
ops = sys.argv[1].split(";;")
for op in ops:
    op = op.strip()
    if op:
        subprocess.run(["python3", "/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/orch/cl.py"] + shlex.split(op), check=True)
PY
  git add CHANGELOG.md
  for i in 1 2 3; do git commit -q --amend --no-edit > $SP/orch/amend_out.txt 2>&1 && break; git add -A pybads CHANGELOG.md; done
fi
git log --oneline -1
