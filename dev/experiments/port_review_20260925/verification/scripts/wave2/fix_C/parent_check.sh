#!/bin/bash
# Run the worktree's current test file against the package at HEAD (the
# parent of the commit being prepared). Usage: parent_check.sh <test file> <k expr>
S=/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/fix/C
W=/home/user/pybads-fix-C
rm -rf $S/parent && mkdir -p $S/parent
cd $W && git archive HEAD pybads | tar -x -C $S/parent
cp $W/$1 $S/parent/$1
cd $S/parent && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$S/parent:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -m pytest -p no:cacheprovider $1 -q -k "$2" 2>&1
