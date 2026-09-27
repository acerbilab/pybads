#!/bin/bash
# The rest of wave 4's fix pass after batch 2: retag the periodic_vars pick, W4-29, W4-30, W4-1 and its gate, W4-6 and its gates.
S=/tmp/claude-0/-home-user-pybads/6250b50f-b045-5c7a-88e2-48abf62425c8/scratchpad
cd /home/user/pybads
set -o pipefail
# 1. retag the periodic_vars pick (HEAD at the end of batch 2)
git log -1 --format=%s | grep -q "(W4, periodic_vars)" || { echo "HEAD is not the periodic_vars pick"; exit 1; }
git log -1 --format=%B | sed '1s/(W4, periodic_vars)/(W4, found while verifying)/' | git commit -q --amend -F - || exit 1
echo "retagged: $(git log -1 --format='%h %s' | cut -c1-100)"
# 2. W4-29
$S/orch/picks_run.sh $S/orch/w429.list || { echo "STOP at W4-29"; exit 1; }
# 3. W4-30, the orchestrator's docstring commit
python3 $S/orch/w430_edit.py || exit 1
.venv/bin/pre-commit run --files pybads/bads/optimize_result.py > /dev/null 2>&1
git add pybads/bads/optimize_result.py
git commit -q -F $S/orch/msg_w430.txt || { git add pybads/bads/optimize_result.py; git commit -q -F $S/orch/msg_w430.txt || exit 1; }
echo "$(git log -1 --format='%h %s' | cut -c1-100)"
$S/orch/picks_run.sh /dev/null --suite-head || { echo "STOP at W4-30"; exit 1; }
# 4. W4-1 and its gate against W4-21's population
$S/orch/picks_run.sh $S/orch/w41.list || { echo "STOP at W4-1"; exit 1; }
h41=$(git log -1 --format=%h --fixed-strings --grep="(W4-1)")
$S/orch/gate.sh w41 $h41 default dev/scripts/runs/population/w421_86512c9 w421 &
g41=$!
# 5. W4-6, picked while W4-1's gate runs
$S/orch/picks_run.sh $S/orch/w46.list || { echo "STOP at W4-6"; wait $g41; exit 1; }
h46=$(git log -1 --format=%h --fixed-strings --grep="(W4-6)")
wait $g41
grep -q "no configuration flagged" $S/gates/w41_vs_w421.md || { echo "W4-1 FLAGGED: stop before W4-6's gate"; exit 1; }
$S/orch/gate.sh w46 $h46 default dev/scripts/runs/population/w41_$h41 w41
$S/orch/gate.sh geo_w46 $h46 geometry dev/scripts/runs/population/geo_w421_86512c9 geo_w421
echo "chain done: W4-1 $h41, W4-6 $h46"
