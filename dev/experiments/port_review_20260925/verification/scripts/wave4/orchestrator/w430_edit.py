p = "/home/user/pybads/pybads/bads/optimize_result.py"
s = open(p).read()
old = """        - fsd: float
            - Standard deviation of objective function at solution (0 if noiseless).
"""
new = """        - fsd: float
            - Standard deviation of objective function at solution (0 if noiseless).
              For a noisy run that ``output_fcn`` stops in its
              initialization, which takes no final samples, it is not an
              estimate: ``noise_size`` without ``specify_target_noise``,
              and otherwise the standard deviation that the target returned
              at the incumbent.
"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("ok")
