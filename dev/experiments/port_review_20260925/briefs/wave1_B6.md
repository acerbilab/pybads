# Slice part: B6, the GP model and its gpyreg objects

Title of the report: `# B6 <track> review: GP model and its gpyreg objects`, with `<track>` "internal" or "comparison".

## The slice

Your slice is **B6, the GP model**: the kernel, the mean and the noise that PyBADS builds with gpyreg, their hyperparameters in the units each side uses, their hyperpriors and bounds and the updates of these at each rebuild, and the prediction as PyBADS asks for it. The training set and the policy of refits (which points, when to refit, how a fit is retried) are slice B5, reviewed at the same time by others: read them where you need them, report on them only where the model depends on them.

Python (`{PYBADS_REVIEW}`):
- `pybads/bads/gaussian_process_train.py`: `init_and_train_gp`, `_gp_hyp`, the prior and bound updates in `local_gp_fitting` (the mean, the noise, the covariance), `_meanfun_name_to_mean_function`, `_cov_identifier_to_covariance_function`, `_get_random_samples_from_priors_`, `_get_samples_from_slice_sampler_`, and the call of the name-mangled private `gp._GP__gp_obj_fun`;
- `pybads/stats/get_hpd.py` (read only by `_gp_hyp`);
- in `{GPYREG}/gpyreg/`, as PyBADS configures and calls them: `RationalQuadraticARD` (`covariance_functions.py`), `ConstantMean` and `NegativeQuadratic` (`mean_functions.py`), `GaussianNoise` (`noise_functions.py`) with the flags PyBADS sets, and `GP.fit`, `GP.update`, `GP.predict`, `GP.set_priors`, `GP.set_bounds` in `gaussian_process.py`, including how a prior truncated by the bounds is normalized;
- where PyBADS predicts from the GP (grep `predict(` in `pybads/bads/bads.py`, `pybads/search/`, `pybads/acquisition_functions/`), for what it asks: the latent function or an observation, with which variance.

MATLAB counterparts (`{BADS}`), for the comparison track:
- `gpdef/gpdefBads.m` (the whole file: the definition of the GP and the updates of its hyperpriors at each training);
- `gpml_fast/covRQard_fast.m` (the kernel and its derivatives), `gpml_fast/infPrior_fast.m`, `gpml_fast/infExact_fastrobust.m`, `gpml_fast/sq_dist_fast.m`;
- `utils/likGaussHe.m` (the likelihood with the noise the target returns), `utils/mygp.m`, `utils/gppred.m`, `utils/gppriorrnd.m`, `utils/gpset.m`, `utils/prctile1.m`, `utils/minimizebnd.m` as far as the bounds go;
- in GPML (`gpml-matlab-v3.6-2015-07-07/`), as the reference for the priors, the mean and the factorization: `priorGauss`, `meanConst`, `sq_dist`, `solve_chol`, and the covariance conventions of `covRQard`;
- unused by MATLAB's defaults: `gpdef/private/gpdefStationaryNew.m`, `gpml_fast/exact_inference_*.m`, `gpml_fast/infExact_fast.m`, `utils/private/fminbayes.m`.

## How PyBADS reaches this code at default options

PyBADS hard-wires the rational-quadratic ARD kernel (`optim_state["gp_cov_fun"] = 1`), a constant mean, and Gaussian noise: a fitted constant, plus at uncertainty level 2 (`specify_target_noise`) the variances the target returns. `_gp_hyp` sets the hyperpriors and bounds on the initial design; `local_gp_fitting` updates the priors at each rebuild. Hyperparameters are optimized at default options (`gp_samples = 0`). Say for every finding whether a default run reaches it, at which uncertainty level, and which option or input reaches it otherwise. gpyreg's `RationalQuadraticARD` has no counterpart in gpyreg's own MATLAB reference (`gplite`), and no other user of gpyreg reaches it: its reference here is GPML's `covRQard`.

## First questions

Answer each under its own heading:
1. **The kernel.** Does gpyreg's `RationalQuadraticARD`, with the hyperparameters PyBADS gives it, compute the kernel of GPML's `covRQard` as BADS calls it: the parameterization of the length scales, the output scale and the shape, the logarithms, and the derivatives with respect to each hyperparameter (the fit uses them)? Check on random inputs, against an independent implementation or finite differences.
2. **The hyperpriors and bounds.** For every hyperparameter (length scales, output scale, shape, noise, mean), are the prior of `_gp_hyp` and its update at each rebuild the prior of `gpdefBads.m`, as a distribution in the same units (type, centre, width), and are the bounds the same? In particular the constant mean: its bounds, which PyBADS sets from the initial design, and its prior, re-centred at each rebuild: can the prior fall outside the bounds, and what does gpyreg then compute for the log prior and its normalization? On the internal track: are the priors and bounds what the comments and the paper say, and consistent with each other?
3. **The noise and the prediction.** At uncertainty level 2, does the GP's noise equal `likGaussHe`'s (the target's variances plus a fitted constant, in which units), and at levels 0 and 1 the fitted constant alone? Where PyBADS predicts, does it get what MATLAB's `gppred` gives at the same call (the latent mean and variance, or those of an observation)?

---

The rest of the prompt is `wave1_common.md` (from "You are a reviewer") and the part of your track.
