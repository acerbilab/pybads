#!/bin/bash
# Replays the committed cl.py's operations of each pick on its parent's CHANGELOG.md and diffs with the pick's.
R=/home/user/pybads-review
O=$R/dev/experiments/port_review_20260925/verification/scripts/wave4/orchestrator
E=$O/changelog_entries
D=/home/user/dc4/d_records/cltest
cl() { python3 $D/cl.py "$@" > /dev/null || echo "cl.py failed: $*"; }
check() { # commit, then ops as a function
  c=$1; shift
  git -C $R show $c^:CHANGELOG.md > $D/CHANGELOG.md
  "$@"
  git -C $R show $c:CHANGELOG.md > $D/pick.md
  if diff -q $D/CHANGELOG.md $D/pick.md > /dev/null; then echo "$c: replay identical"; else echo "$c: replay DIFFERS"; diff $D/CHANGELOG.md $D/pick.md | head -20; fi
}
op_8daf7ad() { cl fixed $E/w44_fixed.txt; cl upgrading $E/w44_up.txt; }
op_5dd92b7() { cl fixed $E/w48_fixed.txt; cl upgrading $E/w48_up.txt; }
op_29a258a() { cl fixed $E/w49_fixed.txt; }
op_86512c9() { cl replace_bullet "**Points evaluated again.**" $E/w421_points.txt; cl replace_bullet "**Scale of the evolution-strategy search.**" $E/w421_scale.txt; }
op_6e24519() { cl changed $E/w418_changed.txt; cl upgrading $E/w418_up.txt; }
op_36c9ec1() { cl replace_bullet "**LCB parameter of the search.**" $E/w419_lcb.txt; cl replace_bullet "A \`search_acq_fcn\` whose" $E/w419_up.txt; }
op_36e8b70() { cl replace_bullet "**Checks of" $E/w425_checks.txt; cl upgrading $E/w425_up.txt; }
op_6f673a2() { cl replace_bullet "**Search without a candidate.**" $E/w427_search.txt; }
op_b61a880() { cl replace_bullet "**Noisy runs that end within" $E/w414_noisy.txt; }
op_6c36782() { cl replace_bullet "**Sto-BADS poll.**" $E/w415_sto.txt; }
op_b78f782() { cl fixed $E/pv_fixed.txt; }
op_bd793f2() { cl replace_bullet "**\`hedge_gamma\`.**" $E/w429_hedge.txt; cl replace_bullet "\`BADS\` raises \`ValueError\` for a \`hedge_gamma\`" $E/w429_up.txt; }
op_efe5e95() { cl fixed $E/w41_fixed.txt; }
op_e7bd01d() { cl fixed $E/w46_fixed.txt; }
op_46af65a() { cl replace "Noise test in the schedule of the GP's fits." $E/w46c_noise.txt; }
op_fba29cd() { cl replace "Repeated points with user-specified noise." $E/w45_repeats.txt; }
for c in 8daf7ad 5dd92b7 29a258a 86512c9 6e24519 36c9ec1 36e8b70 6f673a2 b61a880 6c36782 b78f782 bd793f2 efe5e95 e7bd01d 46af65a fba29cd; do check $c op_$c; done
