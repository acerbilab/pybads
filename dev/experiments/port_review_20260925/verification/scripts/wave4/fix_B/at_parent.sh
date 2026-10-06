#!/bin/bash
# Usage: at_parent.sh <name> <rev> <test file relative path> [pytest args...]
# Extracts the package at <rev> into scratch, copies the worktree's version of
# the test file over it, and runs pytest there.
set -e
name=$1; rev=$2; tf=$3; shift 3
S=/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad/wave4/fix_B/parent_$name
rm -rf "$S"; mkdir -p "$S"
cd /home/user/pybads-fix-B
git archive "$rev" pybads | tar -x -C "$S"
mkdir -p "$S/$(dirname $tf)"
touch "$S/$(dirname $tf)/__init__.py"
cp "/home/user/pybads-fix-B/$tf" "$S/$tf"
cd "$S"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$S:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -m pytest "$tf" -q -p no:cacheprovider "$@" 2>&1 | tail -40
