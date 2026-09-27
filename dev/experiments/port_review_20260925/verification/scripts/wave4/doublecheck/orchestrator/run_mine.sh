#!/bin/bash
# The orchestrator's own checks of the doublecheck of wave 4: the suite at 81385ac, then the fingerprints.
cd /home/user/dc4/orchestrator
echo "suite start $(date -u +%H:%M:%S)"
(cd /home/user/pybads-fp && PYTHONPATH="/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -c "import pybads, gpyreg, numpy, scipy, sys; print(sys.version.split()[0], numpy.__version__, scipy.__version__, pybads.__file__, gpyreg.__file__)") > suite_81385ac.log 2>&1
(cd /home/user/pybads-fp && git log -1 --format='%h' && PYTHONPATH="/home/user/gpyreg-v1.3.3" /home/user/pybads/.venv/bin/python -m pytest -p no:cacheprovider -vv 2>&1) >> suite_81385ac.log
echo "suite end $(date -u +%H:%M:%S)"
./fp_dc.sh 1 8c8d6f8 86512c9 4b84a2d efe5e95 e7bd01d 46af65a 81385ac > fp_dc.out 2>&1
./fp_dc.sh default 8c8d6f8 86512c9 4b84a2d efe5e95 e7bd01d 46af65a 81385ac >> fp_dc.out 2>&1
echo "fp end $(date -u +%H:%M:%S)"
