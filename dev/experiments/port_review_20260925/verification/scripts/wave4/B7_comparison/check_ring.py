# Transcription of funlogger.m:120-121 (1-based rows), nmax = 5
import gpyreg

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
nmax, Xn, Xmax, rows = 5, 0, 0, []
for k in range(1, 10):
    Xn = max(1, (Xn + 1) % nmax)
    Xmax = min(Xmax + 1, nmax)
    rows.append(Xn)
print("row written by evaluations 1..9:", rows, "final Xmax", Xmax, flush=True)
