"""Print the options whose (key, value, description) differ between two
checkouts' _read_config_file."""
import importlib.util
import sys


def load(root):
    spec = importlib.util.spec_from_file_location(
        "opts_" + str(abs(hash(root))), root + "/pybads/bads/options.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old, new = load(sys.argv[1]), load(sys.argv[2])
for f in ["basic_bads_options.ini", "advanced_bads_options.ini"]:
    p = sys.argv[2] + "/pybads/bads/option_configs/" + f
    a = old._read_config_file(p)
    b = new._read_config_file(p)
    assert len(a) == len(b), (len(a), len(b))
    for ra, rb in zip(a, b):
        assert ra[0] == rb[0] and ra[1] == rb[1], (ra, rb)
        if ra[2] != rb[2]:
            print(f"{ra[0]}:\n  old: {ra[2]}\n  new: {rb[2]}")
