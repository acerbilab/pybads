"""Edit CHANGELOG.md's Unreleased section.
cl.py fixed|changed|upgrading FILE   : append the entry in FILE to that list
cl.py extend "TITLE" FILE           : append FILE's text to the entry **TITLE** (a new paragraph line joined)"""
import sys
import textwrap

P = "/home/user/pybads/CHANGELOG.md"
s = open(P).read()
head, rest = s.split("## [Unreleased]\n", 1)
unrel, tail = rest.split("\n## [1.1.0]", 1)
mode = sys.argv[1]


def wrap_entry(text):
    text = " ".join(text.split())
    return textwrap.fill(
        text,
        76,
        initial_indent="- ",
        subsequent_indent="  ",
        break_on_hyphens=False,
        break_long_words=False,
    )


if mode in ("fixed", "changed", "upgrading"):
    entry = wrap_entry(open(sys.argv[2]).read())
    marker = {
        "upgrading": "\n### Changed\n",
        "changed": "\n### Fixed\n",
        "fixed": None,
    }[mode]
    if marker is None:
        unrel = unrel.rstrip("\n") + "\n" + entry + "\n"
    else:
        a, b = unrel.split(marker, 1)
        unrel = a.rstrip("\n") + "\n" + entry + "\n" + marker + b
elif mode == "extend":
    title, add = sys.argv[2], " ".join(open(sys.argv[3]).read().split())
    start = unrel.index(f"- **{title}**")
    end = unrel.find("\n- ", start + 1)
    nxt_h = unrel.find("\n### ", start + 1)
    ends = [e for e in (end, nxt_h, len(unrel.rstrip("\n"))) if e != -1]
    end = min(ends)
    seg = unrel[start:end]
    end = start + len(seg.rstrip("\n"))
    lines = unrel[start:end].split("\n")
    last = lines[-1]
    indent = "- " if len(lines) == 1 else "  "
    tail_txt = " ".join((last[2:] + " " + add).split())
    wrapped = textwrap.fill(
        tail_txt,
        76,
        initial_indent=indent,
        subsequent_indent="  ",
        break_on_hyphens=False,
        break_long_words=False,
    )
    unrel = unrel[:start] + "\n".join(lines[:-1] + [wrapped]) + unrel[end:]
else:
    sys.exit("mode?")
open(P, "w").write(head + "## [Unreleased]\n" + unrel + "\n## [1.1.0]" + tail)
print("ok", mode)
