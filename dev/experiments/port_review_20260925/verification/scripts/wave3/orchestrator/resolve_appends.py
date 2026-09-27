"""resolve_appends.py FILE...: resolve diff3 conflict hunks whose base is empty (both sides added lines at the same place) by keeping ours, then theirs. Exit 1 if a hunk has a non-empty base."""
import sys

ok = True
for path in sys.argv[1:]:
    lines = open(path).read().split("\n")
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("<<<<<<< "):
            ours, base, theirs, part = [], [], [], "ours"
            i += 1
            while not lines[i].startswith(">>>>>>> "):
                if lines[i].startswith("||||||| "):
                    part = "base"
                elif lines[i] == "=======":
                    part = "theirs"
                else:
                    {"ours": ours, "base": base, "theirs": theirs}[
                        part
                    ].append(lines[i])
                i += 1
            if any(b.strip() for b in base):
                ok = False
                print(f"{path}: a hunk with a non-empty base")
            out += ours + ["", ""] + theirs
        else:
            out.append(l)
        i += 1
    open(path, "w").write("\n".join(out))
sys.exit(0 if ok else 1)
