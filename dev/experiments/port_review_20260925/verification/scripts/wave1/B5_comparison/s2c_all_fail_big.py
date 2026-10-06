"""All ten tries of _robust_gp_fit_ fail, with noise_nudge = [0, 0] so that
the bound nudge does not stop it first: what does the caller receive?"""
exec(open("s2_big.py").read().split("for n_fail in")[0])
options = copy.deepcopy(options)
options["noise_nudge"] = np.array([0, 0])
fit, calls = make_fit(10)
gpr.GP.fit = fit
try:
    out = gpt._robust_gp_fit_(
        copy.deepcopy(gp),
        X.copy(),
        Y.copy(),
        None if s2 is None else s2.copy(),
        hyp_gp.copy(),
        gp_train,
        optim_state,
        options,
        np.random.default_rng(0),
    )
    print("returned", out[3])
except Exception as e:
    print(f"after {len(calls)} tries: RAISED {type(e).__name__}: {e}")
finally:
    gpr.GP.fit = orig_fit
