#!/bin/bash
# one_run.sh COMMIT LABEL SEED: one run from a worktree at COMMIT; prints the final fval
cd /home/user/pybads
c=$(git rev-parse --short $1); WT=dev/scripts/runs/worktrees/h_$c
[ -d $WT ] || git worktree add --detach $WT $c > /dev/null 2>&1
OUT=/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/bisect/${c}_$2_$3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/user/gpyreg-v1.3.3 .venv/bin/python -u $WT/dev/scripts/population.py run --suite default --only $2 --seeds $3 --workers 1 --out $OUT > $OUT.log 2>&1
python3 -c "import json;d=json.load(open('$OUT/$2_seed$3.json'));print('$c', d['final']['fval'], d['final']['func_count'])"
