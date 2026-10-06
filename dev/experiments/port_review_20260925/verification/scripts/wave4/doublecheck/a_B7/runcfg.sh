#!/bin/bash
# usage: runcfg.sh REV LABEL MODE SEEDS OUT
cd /home/user/dc4/a_B7
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="/home/user/dc4/a_B7/at_$1:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -u runcfg.py "$2" "$3" "$4" "$5" 2>&1 | grep -v "Warning\|warn(\|^ \+[a-z_]"
