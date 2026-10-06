# The fingerprints of the doublecheck of wave 4 on Windows (PowerShell), for the table of
# verification/wave4.md, "Doublecheck": dev/scripts/fingerprint.py at the pass's key commits,
# with the default number of BLAS threads and with one, gpyreg from the clone at v1.3.3.
# Set the three paths, then run this file from any directory: .\fp_windows.ps1
$repo   = "C:\path\to\pybads"            # the checkout with .venv (dev/scripts/runs/LOCAL.md)
$gpyreg = "C:\path\to\gpyreg-v1.3.3"     # a gpyreg clone checked out at the tag v1.3.3
$wt     = "C:\path\to\pybads-fp"         # a detached worktree for the fingerprints, made below

git -C $repo fetch origin dev-next dev-port-review-w4
if (-not (Test-Path $wt)) { git -C $repo worktree add --detach $wt 81385ac }
$commits = "8c8d6f8", "86512c9", "4b84a2d", "efe5e95", "e7bd01d", "46af65a", "81385ac"
$env:PYTHONPATH = "$wt;$gpyreg"
foreach ($threads in "default", "1") {
    if ($threads -eq "1") {
        $env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
    } else {
        Remove-Item Env:OMP_NUM_THREADS, Env:OPENBLAS_NUM_THREADS, Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
    }
    foreach ($c in $commits) {
        git -C $wt checkout -q --detach $c
        Push-Location $wt
        $fp = & "$repo\.venv\Scripts\python.exe" -u dev\scripts\fingerprint.py 2>&1 | Select-Object -Last 1
        Pop-Location
        "$c | threads=$threads | $fp"
    }
}
& "$repo\.venv\Scripts\python.exe" -c "import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"
# Each line ends with pybads.__file__, which must lie under $wt, and the hash.
# Afterwards: git -C $repo worktree remove $wt
