p = "pybads/init_functions/init_sobol.py"
s = open(p).read()
old_sig = """def init_sobol(
    u0=np.ndarray,
    lb=np.ndarray,
    ub=np.ndarray,
    plb=np.ndarray,
    pub=np.ndarray,
    fun_eval_start=int,
    rng=None,
):
"""
new_sig = """def init_sobol(
    u0,
    lb,
    ub,
    plb,
    pub,
    fun_eval_start,
    rng=None,
):
"""
assert old_sig in s
s = s.replace(old_sig, new_sig)
old_doc = """    Parameters
    ----------
    u0 : array_like
        Initial point.
    lb : array_like
        Lower bounds.
    ub : array_like
        Upper bounds.
    plb : array_like
        Lower bounds for the parameters.
    pub : array_like
        Upper bounds for the parameters.
    fun_eval_start : int
        Number of initial function evaluations.
"""
new_doc = """    The design has ``2**ceil(log2(fun_eval_start))`` points, twice as many
    when that number equals the dimension ``D``, scaled to the plausible box.

    Parameters
    ----------
    u0 : np.ndarray
        The starting point, of shape ``(D,)``.
    lb : np.ndarray
        The lower bounds (unused).
    ub : np.ndarray
        The upper bounds (unused).
    plb : np.ndarray
        The plausible lower bounds, which the design spans.
    pub : np.ndarray
        The plausible upper bounds, which the design spans.
    fun_eval_start : int
        The number of points asked of the design, which is rounded up as
        above.
"""
assert old_doc in s
s = s.replace(old_doc, new_doc)
old_ret = """    Returns
    -------
    u_init : array_like
        Initial points.
    n_samples : int
        Number of samples used for the initialization.
"""
new_ret = """    Returns
    -------
    u_init : np.ndarray
        The points of the design, of shape ``(n_samples, D)``.
    n_samples : int
        The number of points of the design.
"""
assert old_ret in s
s = s.replace(old_ret, new_ret)
old_code = """    n_samples = int(np.ceil(np.log2(fun_eval_start)))
    if 2**n_samples == u0.size:
        n_samples += 1
    samples = sobol_sampler.random_base2(n_samples)

    u_init = plb + samples * (pub - plb)

    return u_init, n_samples
"""
new_code = """    m = int(np.ceil(np.log2(fun_eval_start)))
    if 2**m == u0.size:
        m += 1
    samples = sobol_sampler.random_base2(m)
    n_samples = samples.shape[0]

    u_init = plb + samples * (pub - plb)

    return u_init, n_samples
"""
assert old_code in s
s = s.replace(old_code, new_code)
open(p, "w").write(s)
