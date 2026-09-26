"""B3-K12 (search_n_try type), B3-K7 (_update_search_stats_ vs UpdateSearch), B3-K5 (ESSearchCMA)."""
from types import SimpleNamespace

import numpy as np
import vhdr  # noqa
from capture import capture_states

from pybads import BADS
from pybads.bads.option_configs import get_pybads_option_dir_path
from pybads.bads.options import Options
from pybads.search.es_search import ESSearchCMA


def load(D):
    p = get_pybads_option_dir_path()
    o = Options(
        p + "/basic_bads_options.ini",
        evaluation_parameters={"D": D},
        user_options=None,
    )
    o.load_options_file(
        p + "/advanced_bads_options.ini", evaluation_parameters={"D": D}
    )
    return o


print(
    "B3-K12 search_n_try:",
    [
        (D, load(D)["search_n_try"], type(load(D)["search_n_try"]).__name__)
        for D in (1, 2, 3, 6, 7, 20)
    ],
)
print(
    "         MATLAB SearchNtry = max(nvars, floor(3+nvars/2)):",
    [(D, max(D, int(np.floor(3 + D / 2)))) for D in (1, 2, 3, 6, 7, 20)],
)


# B3-K7: transcription of UpdateSearch (bads.m:1342-1375)
def update_search_matlab(os_, status, dist, opt):
    os_.setdefault("stats", {"lsf": [], "succ": [], "udist": []})
    os_["stats"]["lsf"].append(np.log(os_["sf"]))
    os_["stats"]["udist"].append(dist)
    if status == "success":
        os_["stats"]["succ"].append(1)
        os_["sf"] *= opt["search_scale_success"]
        if opt["adaptive_incumbent_shift"]:
            os_["sd"] *= 2
    elif status == "incremental":
        os_["stats"]["succ"].append(0.5)
        os_["sf"] *= opt["search_scale_incremental"]
        if opt["adaptive_incumbent_shift"]:
            os_["sd"] *= 4
    elif status == "failure":
        os_["stats"]["succ"].append(0)
        os_["sf"] = max(
            opt["search_factor_min"], os_["sf"] * opt["search_scale_failure"]
        )
        if opt["adaptive_incumbent_shift"]:
            os_["sd"] = max(opt["incumbent_sigma_multiplier"], os_["sd"] / 2)
    if os_["count"] == opt["search_n_try"]:
        os_["sf"] = 1


rng = np.random.default_rng(0)
for ais in (False, True):
    opt = dict(load(3))
    opt["adaptive_incumbent_shift"] = ais
    fake = SimpleNamespace(
        options=opt,
        optim_state={
            "search_factor": 1,
            "sd_level": opt["incumbent_sigma_multiplier"],
        },
    )
    m = {"sf": 1, "sd": opt["incumbent_sigma_multiplier"]}
    mism = 0
    n = 0
    for k in range(400):
        cnt = (k % opt["search_n_try"]) + 1
        s = rng.choice(
            ["success", "incremental", "failure", "failure", "failure"]
        )
        fake.optim_state["search_count"] = cnt
        m["count"] = cnt
        BADS._update_search_stats_(fake, s, 0.1)
        update_search_matlab(m, s, 0.1, opt)
        n += 1
        mism += not (
            np.isclose(fake.optim_state["search_factor"], m["sf"])
            and np.isclose(fake.optim_state["sd_level"], m["sd"])
        )
    st = fake.optim_state["search_stats"]
    same_stats = np.allclose(
        st["log_search_factor"], m["stats"]["lsf"]
    ) and np.allclose(st["success"], m["stats"]["succ"])
    print(
        f"B3-K7 adaptive_incumbent_shift={ais}: {mism}/{n} mismatches of search_factor/sd_level; stats lists equal: {same_stats}; "
        f"min factor reached {np.exp(min(st['log_search_factor'])):.4f} (search_factor_min={opt['search_factor_min']})"
    )

# B3-K5: ESSearchCMA called directly
st = capture_states(D=3, seed=0, max_fun_evals=40, max_states=1)[0]
try:
    es = ESSearchCMA(2048, 2048, st["options"], rng=np.random.default_rng(0))
    es(
        st["u"],
        None,
        None,
        st["func_logger"],
        st["gp"],
        st["optim_state"],
        True,
        None,
    )
    print("B3-K5: ESSearchCMA ran")
except Exception as e:
    import traceback

    tb = traceback.extract_tb(e.__traceback__)[-1]
    print(
        f"B3-K5: ESSearchCMA raises {type(e).__name__}: {e} (at line {tb.lineno}: {tb.line})"
    )
from pybads.search.search_hedge import ESSearchHedge

try:
    h = ESSearchHedge(
        [("ES-cma+", 1)], st["options"], rng=np.random.default_rng(0)
    )
    h(st["u"], None, None, st["func_logger"], st["gp"], st["optim_state"])
except Exception as e:
    print(f"B3-K5: the hedge given 'ES-cma+' raises {type(e).__name__}: {e}")
