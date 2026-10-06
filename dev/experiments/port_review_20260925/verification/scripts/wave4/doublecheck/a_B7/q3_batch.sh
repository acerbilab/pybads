#!/bin/bash
# ellipsoid_D3_homo: 60 full runs in all, one at a time, one BLAS thread
cd /home/user/dc4/a_B7
./runcfg.sh 86512c9 ellipsoid_D3_homo asis 0-3 edh_w421.jsonl
./runcfg.sh efe5e95 ellipsoid_D3_homo force:967 0-29 edh_w41_force967.jsonl
./runcfg.sh efe5e95 ellipsoid_D3_homo asis 0,1 edh_w41.jsonl
./runcfg.sh efe5e95 ellipsoid_D3_homo forcedraw:967 0-23 edh_w41_forcedraw967.jsonl
echo BATCH DONE
