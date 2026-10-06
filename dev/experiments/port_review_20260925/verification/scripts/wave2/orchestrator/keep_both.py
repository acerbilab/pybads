"""keep_both.py FILE: resolve one appended-at-the-end conflict by keeping ours, then theirs."""
import sys

p = sys.argv[1]
s = open(p).read()
a = s.index("<<<<<<< HEAD\n")
m = s.index("\n=======\n", a)
e = s.index("\n>>>>>>> ", m)
e_end = s.index("\n", e + 1) + 1
ours = s[a + len("<<<<<<< HEAD\n") : m].rstrip("\n")
theirs = s[m + len("\n=======\n") : e].strip("\n")
s = s[:a] + ours + "\n\n\n" + theirs + "\n" + s[e_end:]
assert "<<<<<<<" not in s and ">>>>>>>" not in s
open(p, "w").write(s)
print("resolved", p)
