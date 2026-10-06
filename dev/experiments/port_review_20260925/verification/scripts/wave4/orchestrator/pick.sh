#!/bin/bash
# usage: pick.sh COMMIT [cl-op args...]   (cl-op: "fixed FILE", "changed FILE", "upgrading FILE", "extend TITLE FILE"; several separated by ';;')
# Cherry-picks COMMIT onto the current branch of the main checkout and, with
# changelog operations, amends the pick with the changelog lines.
SP=/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad
cd /home/user/pybads
c=$1; shift
if ! git -c merge.conflictStyle=diff3 cherry-pick $c > $SP/orch/pick_out.txt 2>&1; then
  u=$(git diff --name-only --diff-filter=U)
  bad=$(echo "$u" | grep -v '^pybads/testing/')
  if [ -n "$bad" ] || ! python3 $SP/orch/resolve_appends.py $u; then echo "CHERRY-PICK FAILED for $c"; tail -8 $SP/orch/pick_out.txt; git cherry-pick --abort; exit 1; fi
  /home/user/pybads/.venv/bin/pre-commit run --files $u > /dev/null 2>&1; git add $u
  GIT_EDITOR=true git cherry-pick --continue > /dev/null 2>&1 || { echo "CONTINUE FAILED for $c"; exit 1; }
  echo "resolved appends in $u"
fi
ops="$*"
if [ -n "$ops" ]; then
  python3 - "$ops" <<'PY'
import sys, subprocess, shlex
for op in sys.argv[1].split(";;"):
    op = op.strip()
    if op:
        subprocess.run(["python3", "/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad/orch/cl.py"] + shlex.split(op), check=True)
PY
  git add CHANGELOG.md
  for i in 1 2 3; do git commit -q --amend --no-edit > $SP/orch/amend_out.txt 2>&1 && break; git add -A pybads CHANGELOG.md; done
fi
git log --oneline -1
