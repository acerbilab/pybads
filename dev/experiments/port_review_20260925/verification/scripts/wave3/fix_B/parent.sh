#!/bin/bash
# Usage: parent.sh <test-file> <-k expr> [tail lines]
# Extracts HEAD's pybads into $S/parent, copies the worktree's test file over
# it, and runs the test there.
S=/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/fix/B
W=/home/user/pybads-fix-B
rm -rf $S/parent && mkdir -p $S/parent
cd $W && git archive HEAD pybads pyproject.toml | tar -x -C $S/parent
cp $W/$1 $S/parent/$1
cd $S/parent && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$S/parent:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -m pytest $1 -q -p no:cacheprovider -k "$2" 2>&1 | tail -${3:-15}
