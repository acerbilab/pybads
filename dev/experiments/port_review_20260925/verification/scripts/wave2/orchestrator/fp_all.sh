#!/bin/bash
# The fingerprint at every commit of wave 2's fix pass on dev-port-review-w2, from 353ad51 on.
cd /home/user/pybads
WT=dev/scripts/runs/worktrees/h_fp
[ -d $WT ] || git worktree add --detach $WT 353ad51 > /dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for c in 353ad51 $(git rev-list --reverse 353ad51..8510ca8); do
  git -C $WT checkout -q --detach $c
  fp=$(PYTHONPATH="$WT:/home/user/gpyreg-v1.3.3" .venv/bin/python -u dev/scripts/fingerprint.py 2>&1 | tail -1)
  echo "$(git log -1 --format='%h %s' $c | cut -c1-90) | $fp"
done
