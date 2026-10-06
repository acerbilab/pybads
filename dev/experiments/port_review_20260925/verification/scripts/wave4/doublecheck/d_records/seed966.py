import math

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
s = "0.25         -0.5"
codes = [ord(c) for c in s]
p = math.prod(codes)
print("exact", p % 997 + 1, "p", p, "p>2^53", p > 2**53)
# double product (MATLAB prod of uint64 returning double, as the verifier read)
pd = 1.0
for c in codes:
    pd *= float(c)
print("double product", pd, "exact?", int(pd) == p)
# documented formula x - floor(x/y)*y in double
m = pd - math.floor(pd / 997) * 997
print("naive mod", m, "->", m + 1)
print("fmod exact of double", math.fmod(pd, 997) + 1)
