#!/bin/sh
cd /tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/review/B2_verifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="/home/user/pybads-review:/home/user/gpyreg-v1.3.3"
PY=/home/user/pybads/.venv/bin/python
for s in v_budget v_accel v_tolnoise v_options v_funvalues v_overhead v_misc v_actions v_reeval_fail v_neighbors v_thin v_thin2 v_ih_copy; do
  $PY -u $s.py > $s.out 2>&1
done
$PY -u v_move2.py default > v_move2_default.out 2>&1
$PY -u v_move2.py nosearch > v_move2_nosearch.out 2>&1
$PY -u v_move_ab.py > v_move_ab.out 2>&1
echo done > run_all.done
