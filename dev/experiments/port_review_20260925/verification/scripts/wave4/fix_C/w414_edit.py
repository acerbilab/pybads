p = "pybads/bads/bads.py"
s = open(p).read()
old_head = """        # Re-evaluate all best points for noisy evaluations
        yval_vec = self.yval if np.isscalar(self.yval) else self.yval.copy()
        # A run that ends within its first iteration takes no final samples:
        # the result reports the incumbent's observation
        self.optim_state["yval_vec"] = np.atleast_1d(yval_vec).copy()
        self.optim_state["ysd_vec"] = None
        if (
            self.optim_state["uncertainty_handling_level"] > 0
            and poll_iteration > 0
        ):
"""
new_head = """        # Re-evaluate all best points for noisy evaluations
        yval_vec = self.yval if np.isscalar(self.yval) else self.yval.copy()
        # A run that ends in its initialization takes no final samples: the
        # result reports the incumbent's observation
        self.optim_state["yval_vec"] = np.atleast_1d(yval_vec).copy()
        self.optim_state["ysd_vec"] = None
        # The iterate whose point takes the final samples
        final_idx = None
        if (
            self.optim_state["uncertainty_handling_level"] > 0
            and poll_iteration > 0
        ):
"""
assert s.count(old_head) == 1
s = s.replace(old_head, new_head)

start = s.index(
    """            self.best_gp_hyp = self.iteration_history.get("gp_hyp_full")[
                min_q_beta_idx
            ]

            # Re-evalate estimated function value and SD at final point
"""
)
end_marker = """                self.iteration_history.record("fsd", self.fsd, min_q_beta_idx)
"""
end = s.index(end_marker, start) + len(end_marker)
block = s[start:end]
lines = block.split("\n")
# first 3 lines: best_gp_hyp assignment; keep
head = "\n".join(lines[:3]) + "\n"
rest = "\n".join(lines[4:])  # skip blank line
# rest begins with the comment "# Re-evalate ..." then "if nfs > 0:" at 12
rest_lines = rest.split("\n")
assert rest_lines[0].startswith("            # Re-evalate")
assert (
    rest_lines[1] == '            if self.options["noise_final_samples"] > 0:'
)
body = rest_lines[2:]
new_body = []
for l in body:
    if l.strip() == "":
        new_body.append(l)
    else:
        assert l.startswith("                "), l
        new_body.append(l[4:])
new_block = (
    head
    + "            final_idx = min_q_beta_idx\n"
    + """        elif (
            self.optim_state["uncertainty_handling_level"] > 0
            and self.optim_state["iter"] == 0
        ):
            # A run that ends within its first iteration has one iterate,
            # the incumbent, which takes the final samples that the run
            # reserved; MATLAB BADS takes none then (bads.m:1138)
            final_idx = 0

        # Re-evalate estimated function value and SD at final point
        if final_idx is not None and self.options["noise_final_samples"] > 0:
"""
    + "\n".join(new_body)
)
s = s[:start] + new_block + s[end:]
open(p, "w").write(s)
