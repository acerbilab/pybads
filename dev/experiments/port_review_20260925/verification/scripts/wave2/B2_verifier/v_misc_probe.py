"""C-F9 Actions column; K8 poll's returned GP; I-F9 max_iter; I-F10 first
iterations' incumbent value; C-F10 iterations at init; I-F8/C-F13 final message
and yval_vec shape."""
import logging

import numpy as np

from pybads import BADS


class H(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, rec):
        self.msgs.append(rec.getMessage())


class Probe(BADS):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.polls = []
        self.searches = 0
        self.same_gp = []

    def _search_step_(self, gp):
        self.searches += 1
        return super()._search_step_(gp)

    def _poll_step_(self, gp):
        f0, s0 = self.fval, self.fsd
        out = super()._poll_step_(gp)
        self.same_gp.append(out[-1] is gp)
        act = []
        if self.gp_refitted_flag:
            act.append(
                "Train" + (" (failed)" if self.gp_exit_flag < 0 else "")
            )
        if self.last_skipped == self.optim_state["iter"]:
            act.append("skip" if act else "Skip")
        self.polls.append(
            dict(
                it=self.optim_state["iter"],
                shown=self.logging_action[-1],
                matlab=", ".join(act) if act else "",
                fval_in=f0,
                fsd_in=s0,
            )
        )
        return out
