"""Parse the touched docstrings with numpydoc and report warnings and the
parsed parameter names."""
import warnings

import gpyreg
import numpydoc.docscrape as ds

import pybads
from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb
from pybads.bads.bads import BADS
from pybads.function_logger.constraints_check import contraints_check
from pybads.poll.poll_mads_2n import poll_mads_2n
from pybads.search.es_search import ESSearch
from pybads.search.search_hedge import ESSearchHedge

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
for name, obj in [
    ("acq_fcn_lcb", acq_fcn_lcb),
    ("poll_mads_2n", poll_mads_2n),
    ("BADS._get_target_from_gp_", BADS._get_target_from_gp_),
    ("BADS._poll_step_", BADS._poll_step_),
    ("ESSearchHedge.update_hedge", ESSearchHedge.update_hedge),
    ("ESSearch.__call__", ESSearch.__call__),
    ("contraints_check", contraints_check),
]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = ds.FunctionDoc(obj)
    print(f"== {name}: warnings {[str(x.message)[:100] for x in w]}")
    for sec in ("Parameters", "Returns", "Raises"):
        print(f"  {sec}:", [(p.name, p.type) for p in d[sec]])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    d = ds.ClassDoc(BADS)
print("== BADS: warnings", [str(x.message)[:100] for x in w])
print("  Raises:", [(p.name, p.type) for p in d["Raises"]])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    d = ds.ClassDoc(ESSearchHedge)
print("== ESSearchHedge: warnings", [str(x.message)[:140] for x in w])
