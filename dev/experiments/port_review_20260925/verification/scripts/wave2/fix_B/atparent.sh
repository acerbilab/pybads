#!/bin/bash
# usage: atparent.sh <test file relative path> <-k expr>
# Copies HEAD's tree (the parent of the commit to be made), puts the working
# tree's version of the test file in it, and runs the test there.
S=/tmp/claude-0/-home-user-pybads/d2562ae8-2eb6-54ba-872b-e5b256bbba2c/scratchpad/fix/B
W=/home/user/pybads-fix-B
rm -rf $S/parent && mkdir -p $S/parent
(cd $W && git archive HEAD | tar -x -C $S/parent)
cp $W/$1 $S/parent/$1
cd $S/parent
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$S/parent:/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -m pytest $1 -q -k "$2" -p no:cacheprovider -rf 2>&1 | tail -${3:-15}
