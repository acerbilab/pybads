#!/bin/bash
# Run a test file of the worktree against the code of a commit (default HEAD),
# exported to the scratch directory: at_parent.sh <rev> <test file> [pytest args]
S=/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/fix/F
REV=$1; shift
T=$1; shift
D=$S/parent_$REV
rm -rf "$D" && mkdir -p "$D"
git -C /home/user/pybads-fix-F archive "$REV" pybads | tar -x -C "$D"
cp "/home/user/pybads-fix-F/$T" "$D/$T"
cd "$D" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$D:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -m pytest -p no:cacheprovider "$T" -q "$@"
