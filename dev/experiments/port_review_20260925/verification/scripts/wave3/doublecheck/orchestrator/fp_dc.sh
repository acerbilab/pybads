#!/bin/bash
# usage: fp_dc.sh THREADS COMMIT...  fingerprint.py at each commit, from the detached worktree /home/user/pybads-fp.
# THREADS=1 sets OMP/OPENBLAS/MKL_NUM_THREADS=1; THREADS=default leaves them unset.
WT=/home/user/pybads-fp
th=$1; shift
if [ "$th" = 1 ]; then export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; else unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS; fi
for c in "$@"; do
  git -C $WT checkout -q --detach $c
  fp=$(cd $WT && PYTHONPATH="$WT:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -u dev/scripts/fingerprint.py 2>&1 | tail -1)
  echo "$(git -C $WT log -1 --format='%h %s' $c | cut -c1-90) | threads=$th | $fp"
done
