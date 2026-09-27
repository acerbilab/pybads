#!/bin/bash
# usage: run_parent.sh <rev> <test file relative path> [pytest args...]
# Exports pybads/ at <rev> into a scratch tree, copies the worktree's version
# of the test file into it, and runs pytest there.
set -e
S=/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/fix/C
REV=$1; shift; TF=$1; shift
D=$S/parent_tree
rm -rf $D && mkdir -p $D
cd /home/user/pybads-fix-C
git archive $REV pybads pyproject.toml | tar -x -C $D
cp /home/user/pybads-fix-C/$TF $D/$TF
cd $D
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$D:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -m pytest $TF -q -p no:cacheprovider "$@"
