#!/bin/bash
# usage: fp.sh COMMIT...  The fingerprint of dev/scripts/fingerprint.py at each commit, from a detached worktree.
cd /home/user/pybads
WT=/home/user/pybads-fp
[ -d $WT ] || git worktree add --detach $WT HEAD > /dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for c in "$@"; do
  git -C $WT checkout -q --detach $c
  fp=$(cd $WT && PYTHONPATH="$WT:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -u dev/scripts/fingerprint.py 2>&1 | tail -1)
  echo "$(git log -1 --format='%h %s' $c | cut -c1-90) | $fp"
done
