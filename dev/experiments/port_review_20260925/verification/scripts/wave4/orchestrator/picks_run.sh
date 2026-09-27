#!/bin/bash
# usage: picks_run.sh LISTFILE [--suite-head]  Each line: COMMIT|cl-ops. Picks, formats, runs the suite and the fingerprint.
S=/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad
cd /home/user/pybads
suite() {
  h=$(git rev-parse --short HEAD)
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/user/gpyreg-v1.3.3 timeout 900 .venv/bin/python -m pytest -q -p no:cacheprovider -x > $S/orch/suite_$h.full 2>&1
  echo "suite at $h: $(tail -1 $S/orch/suite_$h.full)"
  tail -1 $S/orch/suite_$h.full | grep -q "passed" && ! tail -1 $S/orch/suite_$h.full | grep -q "failed\|error" || { echo "SUITE FAILED at $h"; exit 1; }
  $S/orch/fp.sh $h | tee -a $S/orch/fp_all.out
}
[ "$2" = "--suite-head" ] && suite
while IFS='|' read -r c ops; do
  [ -z "$c" ] && continue
  $S/orch/pick.sh $c "$ops" || { echo "STOP at $c"; exit 1; }
  files=$(git diff-tree --no-commit-id --name-only -r HEAD | grep '\.py$')
  if [ -n "$files" ]; then
    .venv/bin/pre-commit run --files $files > /dev/null 2>&1
    if [ -n "$(git status --porcelain -- $files)" ]; then git add $files; git commit -q --amend --no-edit > /dev/null 2>&1 || { git add $files; git commit -q --amend --no-edit; }; echo "formatted $c"; fi
  fi
  suite
done < $1
echo done
