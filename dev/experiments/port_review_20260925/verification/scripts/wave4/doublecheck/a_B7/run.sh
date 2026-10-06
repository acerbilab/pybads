#!/bin/bash
# usage: ./run.sh script.py [extra PYTHONPATH prefix]
cd /home/user/dc4/a_B7
PP="/home/user/pybads-review:/home/user/gpyreg-v1.3.3"
if [ -n "$2" ]; then PP="$2:/home/user/gpyreg-v1.3.3"; fi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PP" /home/user/pybads/.venv/bin/python -u "$1"
