import sys

sys.path.insert(0, "/home/user/pybads-fix-B")
from pybads.bads.options import _read_config_file

names = set()
for p in [
    "pybads/bads/option_configs/basic_bads_options.ini",
    "pybads/bads/option_configs/advanced_bads_options.ini",
]:
    for k, v, _ in _read_config_file(p):
        if v.split("#")[0].strip() in ("True", "False"):
            names.add(k)
print(len(names))
print(" ".join(sorted(names)))
