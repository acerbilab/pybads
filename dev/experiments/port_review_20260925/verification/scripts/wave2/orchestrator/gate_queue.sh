#!/bin/bash
# The gates of wave 2's moving-results steps, one population at a time, after the baseline (w2-batch1_f6f7f74).
cd /home/user/pybads
P=dev/scripts/runs/population
PY=/home/user/pybads/.venv/bin/python
export PYTHONPATH=/home/user/gpyreg-v1.3.3
run() {  # run SUITE COMMIT NAME
  echo "$(date -u +%H:%M:%S) start $3 ($1 at $2)"
  $PY -u dev/scripts/runs/worktrees/h_$2/dev/scripts/population.py run --suite $1 --seeds 0-29 --workers 4 --out /home/user/pybads/$P/$3 > $P/$3.log 2>&1
  echo "$(date -u +%H:%M:%S) end $3: $(tail -1 $P/$3.log)"
}
cmp() {  # cmp REF NEW
  $PY dev/scripts/population.py compare $P/$1 $P/$2 > $P/cmp_$1__$2.md 2>&1
  echo "compare $1 -> $2: $(grep -ciE 'flag' $P/cmp_$1__$2.md) lines naming a flag"
}
run default 3272bdd w2-16_3272bdd;  cmp w2-batch1_f6f7f74 w2-16_3272bdd
run default c9a2cde w2-29_c9a2cde;  cmp w2-16_3272bdd w2-29_c9a2cde
run default a9fbb97 w2-25_a9fbb97;  cmp w2-29_c9a2cde w2-25_a9fbb97
run bounds a9fbb97 bounds_a9fbb97
run bounds a236eb7 bounds_a236eb7;  cmp bounds_a9fbb97 bounds_a236eb7
run default 8510ca8 w2-head_8510ca8; cmp w2-25_a9fbb97 w2-head_8510ca8
echo "$(date -u +%H:%M:%S) queue done"
