#!/bin/bash
# Run the worktree's current test file(s) against the package at a given commit.
# usage: at_parent.sh <rev> <pytest args...>   (test paths relative to the repo root)
set -e
S=/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad/fix/D
W=/home/user/pybads-fix-D
rev=$1; shift
dir=$S/tree_$rev
rm -rf $dir && mkdir -p $dir
git -C $W archive $rev pybads | tar -x -C $dir
# the tests as they are in the worktree now
cp -r $W/pybads/testing/. $dir/pybads/testing/
cd $dir
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$dir:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -m pytest -p no:cacheprovider "$@"
