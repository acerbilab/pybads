#!/bin/bash
# The gates of the moving-results rows, in the ruled order, each against the step before.
S=/tmp/claude-0/-home-user-pybads/67ddf7b2-c0cb-5e1e-ab93-2e18c75e4a05/scratchpad
L=$S/orch/picks_CD.log
R=/home/user/pybads/dev/scripts/runs/population
h() { grep -oE "^[0-9a-f]{7} fix: .*\($1\)$" $L | tail -1 | cut -c1-7; }
g0=$(grep -o "batch 1 head: [0-9a-f]*" $S/gates/g0.log | cut -d' ' -f4)
es=$(h W3-15); w31=$(h W3-1); w36=$(h W3-6); w319=$(h W3-19); w329=$(h W3-29); w324=$(h W3-24)
echo "g0 $g0 es $es w31 $w31 w36 $w36 w319 $w319 w329 $w329 w324 $w324"
for x in "$g0" "$es" "$w31" "$w36" "$w319" "$w329" "$w324"; do [ -n "$x" ] || { echo "missing hash"; exit 1; }; done
G=$S/orch/gate.sh
$G g1_es $es default $R/g0_batch1_$g0 g0
$G geo_es $es geometry none
$G g2_w31 $w31 default $R/g1_es_$es g1
$G geo_w31 $w31 geometry $R/geo_es_$es geo_es
$G g3_w36 $w36 default $R/g2_w31_$w31 g2
$G g4_w319 $w319 default $R/g3_w36_$w36 g3
$G g5_w329 $w329 default $R/g4_w319_$w319 g4
$G geo_w329 $w329 geometry none
$G g6_w324 $w324 default $R/g5_w329_$w329 g5
$G geo_w324 $w324 geometry $R/geo_w329_$w329 geo_w329
echo "chain done"
