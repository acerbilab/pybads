import numpy as np

import pybads
import pybads.function_logger.function_logger as flm
from pybads.bads import gaussian_process_train as gpt
from pybads.testing.bads.test_run_control import _make_bads

calls = []


class _CountingTimer:
    def start_timer(self, name):
        calls.append(name)

    def stop_timer(self, name):
        pass

    def get_duration(self, name):
        return float(len(calls))


flm.Timer = _CountingTimer
bads = _make_bads(max_fun_evals=10)
bads.optimize()
lg = bads.function_logger
print(pybads.__file__)
print(
    "n_evals[0]",
    lg.n_evals[0, 0],
    "fun_eval_time[0]",
    lg.fun_eval_time[0, 0],
    "sum n_evals",
    np.sum(lg.n_evals[lg.X_flag]),
    "rows",
    lg.Xn + 1,
    "func_count",
    lg.func_count,
    "eff_starting_points",
    bads.optim_state["eff_starting_points"],
)
