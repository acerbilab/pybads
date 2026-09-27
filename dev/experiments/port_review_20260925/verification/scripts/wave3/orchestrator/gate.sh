#!/bin/bash
# usage: gate.sh NAME COMMIT SUITE REFDIR [REFNAME]
# Runs SUITE x seeds 0-29 from a worktree at COMMIT, compares with REFDIR, lists the fields that differ.
S=/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad
name=$1; c=$(git -C /home/user/pybads rev-parse --short $2); suite=$3; ref=$4; refname=${5:-ref}
cd /home/user/pybads
WT=dev/scripts/runs/worktrees/h_$c
[ -d $WT ] || git worktree add --detach $WT $c > /dev/null 2>&1
OUT=dev/scripts/runs/population/${name}_$c
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
start=$(date -u +%H:%M)
PYTHONPATH=/home/user/gpyreg-v1.3.3 .venv/bin/python -u $WT/dev/scripts/population.py run --suite $suite --seeds 0-29 --workers 4 --out $OUT > $S/gates/${name}_run.log 2>&1 || { echo "RUN FAILED $name"; exit 1; }
echo "$name at $c ($suite): $start to $(date -u +%H:%M) UTC"
[ "$ref" = none ] && { PYTHONPATH=/home/user/gpyreg-v1.3.3 .venv/bin/python $WT/dev/scripts/population.py summary $OUT > $S/gates/${name}_summary.md 2>&1; exit 0; }
PYTHONPATH=/home/user/gpyreg-v1.3.3 .venv/bin/python $WT/dev/scripts/population.py compare $ref $OUT > $S/gates/${name}_vs_${refname}.md 2>&1
python3 $S/orch/same_fields.py $ref $OUT > $S/gates/${name}_vs_${refname}_fields.txt 2>&1
grep -i "flag" $S/gates/${name}_vs_${refname}.md | head -5
head -3 $S/gates/${name}_vs_${refname}_fields.txt
