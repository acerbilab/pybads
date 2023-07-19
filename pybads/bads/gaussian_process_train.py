import logging
import math
import traceback
from asyncio.log import logger
from copy import deepcopy

import gpyreg as gpr
import numpy as np
from gpyreg.slice_sample import SliceSampler
from pytest import Function
from scipy.spatial.distance import cdist

from pybads.function_logger import FunctionLogger
from pybads.search.grid_functions import udist
from pybads.stats import get_hpd
from pybads.utils import IterationHistory

from .options import Options


def init_and_train_gp(
    hyp_dict: dict,
    optim_state: dict,
    function_logger: FunctionLogger,
    iteration_history: IterationHistory,
    options: Options,
    plb: np.ndarray,
    pub: np.ndarray,
):
    """
    Initialize and train the Gaussian process model.

    Parameters
    ==========
    hyp_dict : dict
        Hyperparameter summary statistics dictionary.
        If it does not contain the appropriate keys they will be added
        automatically.
    optim_state : dict
        Optimization state from the BADS instance we are calling this from.
    function_logger : FunctionLogger
        Function logger from the BADS instance which we are calling this from.
    iteration_history : IterationHistory
        Iteration history from the BADS instance we are calling this from.
    options : Options
        Options from the BADS instance we are calling this from.
    plb : ndarray, shape (hyp_N,)
        Plausible lower bounds for hyperparameters.
    pub : ndarray, shape (hyp_N,)
        Plausible upper bounds for hyperparameters.

    Returns
    =======
    gp : GP
        The trained GP.
    gp_s_N : int
        The number of samples for fitting.
    sn2hpd : float
        An estimate of the GP noise variance at high posterior density.
    hyp_dict : dict
        The updated summary statistics.
    """

    # Initialize hyp_dict if empty.
    if "hyp" not in hyp_dict:
        hyp_dict["hyp"] = None
    if "warp" not in hyp_dict:
        hyp_dict["warp"] = None
    if "logp" not in hyp_dict:
        hyp_dict["logp"] = None
    if "full" not in hyp_dict:
        hyp_dict["full"] = None
    if "run_cov" not in hyp_dict:
        hyp_dict["run_cov"] = None

    # Get training dataset.
    x_train, y_train, s2_train, t_train = _get_fevals_data(function_logger)
    D = x_train.shape[1]

    # Pick the mean function
    mean_f = _meanfun_name_to_mean_function(optim_state["gp_mean_fun"])

    # Pick the covariance function.
    covariance_f = _cov_identifier_to_covariance_function(
        optim_state["gp_cov_fun"]
    )

    # Pick the noise function.
    const_add = optim_state["gp_noisefun"][0] == 1
    user_add = optim_state["gp_noisefun"][1] == 1
    user_scale = optim_state["gp_noisefun"][1] == 2
    rlod_add = optim_state["gp_noisefun"][2] == 1
    noise_f = gpr.noise_functions.GaussianNoise(
        constant_add=const_add,
        user_provided_add=user_add,
        scale_user_provided=user_scale,
        rectified_linear_output_dependent_add=rlod_add,
    )

    # Setup a GP.
    gp = gpr.GP(D=D, covariance=covariance_f, mean=mean_f, noise=noise_f)
    # Get number of samples and set priors and bounds.
    gp, hyp0, gp_s_N = _gp_hyp(
        optim_state, options, plb, pub, gp, x_train, y_train, function_logger
    )
    # Initial GP hyperparameters.

    if hyp_dict["hyp"] is None:
        hyp_dict["hyp"] = hyp0.copy()

    # Get GP training options.
    gp_train = _get_gp_training_options(
        optim_state,
        iteration_history,
        options,
        hyp_dict,
        gp_s_N,
        function_logger,
    )

    # Build starting points
    hyp0 = np.empty((0, np.size(hyp_dict["hyp"])))

    if gp_train["init_N"] > 0 and optim_state["iter"] > 0:
        # Be very careful with off-by-one errors compared to MATLAB in the
        # range here.
        for i in range(
            math.ceil((np.size(iteration_history["gp"]) + 1) / 2) - 1,
            np.size(iteration_history["gp"]),
        ):
            hyp0 = np.concatenate(
                (
                    hyp0,
                    iteration_history["gp"][i].get_hyperparameters(
                        as_array=True
                    ),
                )
            )
        N0 = hyp0.shape[0]
        if N0 > gp_train["init_N"] / 2:
            hyp0 = hyp0[
                np.random.choice(
                    N0, math.ceil(gp_train["init_N"] / 2), replace=False
                ),
                :,
            ]
    hyp0 = np.concatenate((hyp0, np.array([hyp_dict["hyp"]])))
    hyp0 = np.unique(hyp0, axis=0)

    # In some cases the model can change so be careful.
    if hyp0.shape[1] != np.size(gp.hyper_priors["mu"]):
        hyp0 = None

    fitted = False
    training_failures = 0
    while not fitted:
        try:
            if training_failures == 0:
                _, _, _ = gp.fit(
                    x_train, y_train, s2_train, hyp0=hyp0, options=gp_train
                )
                fitted = True
            elif training_failures == 3:
                # Initialize the hyper-params. to zero after the second failure (like in BADS)
                new_hyp = np.zeros(shape=hyp0.shape)
                _, _, _ = gp.fit(
                    x_train, y_train, s2_train, hyp0=new_hyp, options=gp_train
                )
                hyp0 = new_hyp
                hyp_dict["hyp"] = hyp0
                fitted = True
            else:
                # Sample random from prior
                # In case the initial hyperparameters fails
                new_hyp = _get_random_samples_from_priors_(gp)
                _, _, _ = gp.fit(
                    x_train, y_train, s2_train, hyp0=new_hyp, options=gp_train
                )
                hyp0 = new_hyp
                hyp_dict["hyp"] = hyp0
                fitted = True

        except np.linalg.LinAlgError:
            training_failures += 1
            logger.warning(
                f"bads:gp: Cholesky decomposition has failed. The initial fit on the GP has failed due to the hyp. init."
            )
    # end gp hyp. init

    # Update running average of GP hyperparameter covariance (coarse)
    if hyp_dict["full"] is not None and hyp_dict["full"].shape[1] > 1:
        hyp_cov = np.cov(hyp_dict["full"].T)
        if hyp_dict["run_cov"] is None or options["hyp_run_weight"] == 0:
            hyp_dict["run_cov"] = hyp_cov
        else:
            w = options["hyp_run_weight"] ** options["fun_evals_per_iter"]
            hyp_dict["run_cov"] = (1 - w) * hyp_cov + w * hyp_dict["run_cov"]
    else:
        hyp_dict["run_cov"] = None

    # Missing port: sample for GP for debug (not used)

    # Estimate of GP noise around the top high posterior density region
    # We don't modify optim_state to contain sn2hpd here.
    sn2hpd = _estimate_noise_(gp)

    return gp, gp_s_N, sn2hpd, hyp_dict


def local_gp_fitting(
    gp: gpr.GP,
    current_point,
    function_logger: FunctionLogger,
    options,
    optim_state,
    iteration_history: IterationHistory,
    refit_flag,
):
    """
    Local GP approximation on current point. It updates the priors hyper-parameters and re-fit the Gaussian Process
    """

    # Update the GP training set by setting the NEAREST neighbors (Matlab: gpTrainingSet)
    gp.X, gp.y, s2 = get_grid_search_neighbors(
        function_logger, current_point, gp, options, optim_state
    )
    D = gp.X.shape[1]
    if s2 is not None:
        gp.s2 = s2

    # TODO: Transformation of objective function
    if options["fitness_shaping"]:
        # logger.warn("bads:opt:Fitness shaping not implemented yet")
        pass

    idx_finite_y = np.isfinite(gp.y)
    if np.any(~idx_finite_y):
        y_idx_penalty = np.argmax(gp.y[idx_finite_y])
        gp.y[~idx_finite_y] = gp.y[y_idx_penalty].copy()
        if "S" in optim_state:
            gp.s2[~idx_finite_y] = gp.s2[y_idx_penalty]

    gp.temporary_data["err_y"] = ~idx_finite_y

    # TODO: Rotate dataset (unsupported)

    # Update GP

    ## Update priors hyperparameters using empirical Bayes method.
    if options.get("specify_target_noise"):
        noise_size = options["tol_fun"]  # Additional jitter to specified noise
    else:
        noise_size = options["noise_size"]

    # Update GP Noise
    old_priors = gp.get_priors()
    gp_priors = gp.get_priors()
    prior_noise = gp_priors["noise_log_scale"]
    mu_noise_prior = np.log(noise_size) + options[
        "mesh_noise_multiplier"
    ] * np.log(optim_state["mesh_size"])
    prior_noise = (prior_noise[0], (mu_noise_prior, prior_noise[1][1]))

    # TODO: warped likelihood (unsupported)

    # Update GP Mean
    y_mean = np.percentile(gp.y, options["gp_mean_percentile"])
    y_range = options["gp_mean_range_fun"](y_mean, gp.y)

    prior_mean = None
    if isinstance(gp.mean, gpr.mean_functions.ConstantMean):
        prior_mean = gp_priors["mean_const"]
        prior_mean = (prior_mean[0], (y_mean, prior_mean[1][1]))

    if prior_mean is not None and ~options["gp_fixed_mean"]:
        prior_mean = (prior_mean[0], (prior_mean[1][0], y_range ** (1 / 4)))
    elif options["gp_fixed_mean"]:
        # TODO: update hyp mean by assigning ymean
        pass

    # Update GP Covariance length scale
    if options["gp_cov_prior"] == "iso":
        dist = udist(
            gp.X,
            gp.X,
            1,
            optim_state["lb"],
            optim_state["ub"],
            optim_state["scale"],
            optim_state["periodic_vars"],
        )
        dist = dist.flatten()
        dist = dist[dist != 0]
        if dist.size > 0:
            uu = 0.5 * np.log(np.max(dist))
            ll = 0.5 * np.log(np.min(dist))

            cov_mu = 0.5 * (uu + ll)
            cov_sigma = 0.5 * (uu - ll)

            gp_priors["covariance_log_lengthscale"] = (
                "gaussian",
                (cov_mu, cov_sigma),
            )

    # TODO Adjust prior length scales for periodic variables (mapped to unit circle)

    # Empirical prior on covariance signal variance ((output scale)
    if options["warp_func"] == 0:
        sd_y = np.log(np.std(gp.y))
    else:
        # TODO warp function (Matlab  gpdefbads line-code 302)
        pass

    # Re-fit gaussian Process (optimize or sample -- only optimization supported)
    gp_priors["covariance_log_outputscale"] = ("gaussian", (sd_y, 2.0))
    gp.set_priors(gp_priors)

    old_hyp_gp = gp.get_hyperparameters(as_array=True)
    exit_flag = np.inf
    if refit_flag:
        dic_hyp_gp = gp.get_hyperparameters()
        # In our configuration we have just one sample hyperparameter, in case of multiple we should randomly pick one
        last_dic_hyp_gp = dic_hyp_gp[
            -1
        ]  # Matlab: Initial point #1 is old hyperparameter value (randomly picked)
        # Check for possible high-noise mode
        if (
            np.isscalar(options["noise_size"])
            or len(options["noise_size"] == 1)
            or (
                len(options["noise_size"]) > 1
                and ~np.isfinite(options["noise_size"][1])
            )
        ):
            noise = 1
            is_high_noise = (
                last_dic_hyp_gp["noise_log_scale"]
                > np.log(options["noise_size"]) + 2 * noise
            )
        else:
            noise = options["noise_size"][1]
            is_high_noise = (
                last_dic_hyp_gp["noise_log_scale"]
                > np.log(options["noise_size"][0]) + 2 * noise
            )

        is_low_mean = False  # Check for mean stuck below minimum
        if isinstance(gp.mean, gpr.mean_functions.ConstantMean):
            is_low_mean = last_dic_hyp_gp["mean_const"] < np.min(gp.y)

        # Conditions for performing a second fit
        second_fit = options["double_refit"] | is_high_noise | is_low_mean
        optim_state["second_fit"] = second_fit
        dic_hyp_gp[-1] = last_dic_hyp_gp

        if second_fit:
            # Sample the hyper-params from priors
            prev_hyp_gp = gp.get_hyperparameters(as_array=True)
            if options["use_slice_sampler"]:
                new_hyp = _get_samples_from_slice_sampler_(
                    gp, prev_hyp_gp, optim_state, options
                )
            else:
                new_hyp = _get_random_samples_from_priors_(gp)

            if new_hyp is not None:
                new_hyp = 0.5 * (new_hyp + prev_hyp_gp)
                new_hyp = gp.hyperparameters_to_dict(new_hyp)
                if is_high_noise:
                    new_hyp[0]["noise_log_scale"] = (
                        np.random.randn() - 2
                    )  # Retry with low noise magnitude
                if is_low_mean and isinstance(
                    gp.mean, gpr.mean_functions.ConstantMean
                ):  # Retry with mean from median
                    new_hyp[0]["mean_const"] = np.median(gp.y)

                new_hyp = gp.hyperparameters_from_dict(new_hyp)
                new_hyp = np.minimum(
                    np.maximum(new_hyp, gp.lower_bounds), gp.upper_bounds
                )
                dic_hyp_gp.append(gp.hyperparameters_to_dict(new_hyp)[-1])

        # Matlab: remove undefined points (Not used in BADS, therefore not implemented)
        # Matlab uses hyperSVGD when using multiple samples in the hyp. optimization problem. (Not used in single sample optimization, therefore not implemented)

        gp.set_hyperparameters(dic_hyp_gp, compute_posterior=False)
        hyp_gp = gp.get_hyperparameters(as_array=True)

        # FIT GP
        gp_s_N = 0
        gp_train = _get_gp_training_options(
            optim_state,
            iteration_history,
            options,
            hyp_gp,
            gp_s_N,
            function_logger,
            second_fit=second_fit,
        )
        gp, hyp_gp, res, exit_flag = _robust_gp_fit_(
            gp, gp.X, gp.y, gp.s2, hyp_gp, gp_train, optim_state, options
        )
        dic_hyp_gp = gp.hyperparameters_to_dict(hyp_gp)

        hyp_n_samples = len(dic_hyp_gp)
        # Update after fitting
        # Gaussian process length scale
        if len(dic_hyp_gp[0]["covariance_log_lengthscale"]) > 1:
            len_scale = np.zeros(D)
            for i in range(hyp_n_samples):
                len_scale += len_scale + np.exp(
                    dic_hyp_gp[i]["covariance_log_lengthscale"]
                )
            gp.temporary_data["len_scale"] = len_scale
        else:
            gp.temporary_data["len_scale"] = 1.0

        # GP-based geometric length scale
        ll = np.zeros((hyp_n_samples, D))
        for i in range(hyp_n_samples):
            ll[i, :] = (
                options["gp_rescale_poll"]
                * dic_hyp_gp[i]["covariance_log_lengthscale"]
            )
        # ll = exp(sum(bsxfun(@times, gpstruct.hypweight, ll - mean(ll(:))),2))';
        ll = ll - np.mean(ll)
        ll = np.exp(np.sum(ll, axis=0))

        # Take bounded limits
        ub_bounded = optim_state["ub"].copy()
        ub_bounded[~np.isfinite(ub_bounded)] = optim_state["pub"][
            ~np.isfinite(ub_bounded)
        ]
        lb_bounded = optim_state["lb"].copy()
        lb_bounded[~np.isfinite(lb_bounded)] = optim_state["plb"][
            ~np.isfinite(lb_bounded)
        ]

        ll = np.minimum(
            np.maximum(ll, optim_state["search_mesh_size"]),
            (ub_bounded - lb_bounded) / optim_state["scale"],
        )  # Perhaps this should just be pub - plb?

        gp.temporary_data["poll_scale"] = ll.flatten()

        if options["use_effective_radius"]:
            if isinstance(
                gp.covariance,
                gpr.gpyreg.covariance_functions.RationalQuadraticARD,
            ):
                alpha = np.zeros((hyp_n_samples))
                for i in range(hyp_n_samples):
                    # Casting to a scalar suppress deprecation warnings in pytest runs.
                    # "Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future."
                    alpha[i] = np.exp(dic_hyp_gp[i]["covariance_log_shape"])[0]

                gp.temporary_data["effective_radius"] = np.sqrt(
                    alpha * (np.exp(1 / alpha) - 1)
                )
    else:
        hyp_gp = old_hyp_gp

    # Matlab defines the signal variability in the GP, but is never used.

    # Recompute posterior
    try:
        gp.update(hyp=hyp_gp)
    except np.linalg.LinAlgError:
        # Posterior GP update failed (due to Cholesky decomposition)
        logging.debug(
            "bads:local_gp_fitting: posterior GP update failed. Singular matrix for L Cholesky decomposition"
        )
        gp.set_priors(old_priors)
        gp.set_hyperparameters(old_hyp_gp)
        # gp.set_hyperparameters(iteration_history.get('gp_hyp_full')[-1])
        exit_flag = -2

    return gp, exit_flag


def _meanfun_name_to_mean_function(name: str):
    """
    Transforms a mean function name to an instance of that mean function.

    Parameters
    ==========
    name : str
        Name of the mean function.

    Returns
    =======
    mean_f : object
        An instance of the specified mean function.

    Raises
    ------
    ValueError
        Raised when the mean function name is unknown.
    """
    if name == "zero":
        mean_f = gpr.mean_functions.ZeroMean()
    elif name == "const":
        mean_f = gpr.mean_functions.ConstantMean()
    elif name == "negquad":
        mean_f = gpr.mean_functions.NegativeQuadratic()
    else:
        raise ValueError("Unknown mean function!")

    return mean_f


def _robust_gp_fit_(
    gp: gpr.GP,
    x_train,
    y_train,
    s2_train,
    hyp_gp,
    gp_train,
    optim_state,
    options,
):
    """A private method that compute fit the Gaussian Process.
    In the case it fails to fit the GP with new proposed parameters it sample a new one from the priors.
    """

    noise_nudge = 0

    tmp_gp = deepcopy(gp)
    X = x_train.copy()
    Y = y_train.copy()
    if s2_train is not None:
        s2 = s2_train if np.isscalar(s2_train) else s2_train.copy()
    else:
        s2 = None
    new_hyp = hyp_gp.copy()
    n_try = 10
    success_flag = np.ones((n_try)).astype(bool)
    for i_try in range(0, n_try):
        try:
            new_hyp, _, res = tmp_gp.fit(
                X, Y, s2, hyp0=new_hyp, options=gp_train
            )
            break
        except np.linalg.LinAlgError:
            # handle
            logging.debug(
                "bads:_robust_gp_fit_: posterior GP update failed. Singular matrix for L Cholesky decomposition"
            )
            success_flag[i_try] = False
            if i_try > options["remove_points_after_tries"] - 1:
                idx_drop_out = np.zeros(len(Y)).astype(bool)
                # Remove closest pair sample
                dist = cdist(X, X)
                # Dist is symmetric thus we dont consider the lower triangular and the diagonal of the matrix
                dist[np.tril_indices(dist.shape[0])] = np.inf

                # Indices of the minimum elements
                idx_min = np.unravel_index(
                    np.argmin(dist, axis=None), dist.shape
                )

                if Y[idx_min[0]] > Y[idx_min[1]]:
                    idx_drop_out[idx_min[0]] = True
                else:
                    idx_drop_out[idx_min[1]] = True

                idx_drop_out = np.logical_or(
                    idx_drop_out, (Y > np.percentile(Y, 95)).flatten()
                )
                X = X[~idx_drop_out]
                Y = Y[~idx_drop_out]
                # Remove also user specified noise
                if tmp_gp.s2 is not None and tmp_gp.s2.size > 0:
                    tmp_gp.s2 = tmp_gp.s2[~idx_drop_out]

            # Retry with random sample prior
            old_hyp_gp = (
                hyp_gp.copy() if len(hyp_gp) == 1 else hyp_gp[-1].copy()
            )
            if options["use_slice_sampler"]:

                # if there are multiple hyp samples we take the last one due to the low_mean or high noise.
                if len(new_hyp) > 1:
                    new_hyp = new_hyp[-1].copy()
                new_hyp = _get_samples_from_slice_sampler_(
                    tmp_gp, new_hyp, optim_state, options
                )
            else:
                new_hyp = _get_random_samples_from_priors_(gp)
            if new_hyp is not None:
                new_hyp = 0.5 * (new_hyp + old_hyp_gp)
            else:  # if the slice sampler fail, due to che Cholesky decomposition
                new_hyp = old_hyp_gp

            nudge = options["noise_nudge"]
            if nudge is None or len(nudge) == 0:
                nudge = np.array([0, 0])
            elif len(nudge) == 1:
                nudge = np.vstack((nudge, 0.5 * nudge[0]))

            # Try increase starting point of noise
            noise_nudge = noise_nudge + nudge[0]

            # Increase gp noise hyp lower bounds
            bounds = tmp_gp.get_bounds()
            noise_bound = bounds["noise_log_scale"]
            noise_bound = (noise_bound[0] + noise_nudge, noise_bound[1])
            bounds["noise_log_scale"] = noise_bound
            tmp_gp.set_bounds(bounds)

            # Try increase starting point of noise
            new_hyp = tmp_gp.hyperparameters_to_dict(new_hyp)
            new_hyp[0]["noise_log_scale"] = (
                new_hyp[0]["noise_log_scale"] + noise_nudge
            )
            new_hyp = tmp_gp.hyperparameters_from_dict(new_hyp)
            tmp_gp.set_hyperparameters(new_hyp, compute_posterior=False)

    if np.any(success_flag):
        # at least one run succeeded
        gp.set_hyperparameters(new_hyp, False)
    if np.any(~success_flag):
        # at least one failed
        if options["gp_warnings"]:
            logger.warning(
                f"bads:gpHyperOptFail: Failed optimization of hyper-parameters (after {n_try} attempts). GP approximation might be unreliable."
            )

    if np.all(~success_flag):
        success = -1
    elif np.all(success_flag):
        success = 1
    else:
        success = 0

    return gp, new_hyp, res, success


def _get_random_samples_from_priors_(gp: gpr.GP):
    """
    A private method that retrieves a new set of parameters by randomly sampling from the prior of the GP
    """
    hyp = gp.get_hyperparameters()[-1]  # copy of the hyper-params
    for key, value in gp.get_priors().items():
        if value[0] == "gaussian":
            gauss_parameter = value[1]
            mean_priors = gauss_parameter[0]
            sigma_priors = gauss_parameter[1]
            if "log" in key:
                mean_priors = np.exp(mean_priors)
                sigma_priors = np.exp(sigma_priors)
            new_sample = []
            for idx, m_p in enumerate(mean_priors):
                new_sample.append(np.random.normal(m_p, sigma_priors[idx]))

            hyp[key] = np.array(new_sample)

    return gp.hyperparameters_from_dict(hyp)


def _get_samples_from_slice_sampler_(gp: gpr.GP, hyp_gp, optim_state, options):
    """
    A private method that retrieves a new set of parameters using the slice sampler method.
    """
    hyp_sampler_name = options.get("gp_hyp_sampler", "slicesample")
    if hyp_sampler_name != "slicesample":
        raise ValueError("Wrong sampler")

    sample_f = lambda hyp_: gp._GP__gp_obj_fun(hyp_, False, True)
    width = gp.upper_bounds - gp.lower_bounds

    # If there are multiple parameter samples
    # We reverse and start from the last change, this can happen after a double fit and failing the parameters optimization
    new_hyp = np.flip(np.atleast_2d(hyp_gp), axis=0)
    sampler_failed = True
    for hyp in new_hyp:
        try:
            hyp_sampler = SliceSampler(
                sample_f,
                hyp.flatten(),
                width,
                gp.lower_bounds,
                gp.upper_bounds,
                options={"display": "off", "diagnostics": False},
            )
            new_hyp = hyp_sampler.sample(1, burn=None)["samples"][0]
            sampler_failed = False
            break
        except np.linalg.LinAlgError:
            logger.warning(
                f"bads:gp priors sampling: The slice sampler failed, Cholesky decomposition."
            )
            logger.warning(
                f"bads:gp priors sampling: {traceback.format_exc()}"
            )

    if sampler_failed:
        new_hyp = None
    return new_hyp


def _cov_identifier_to_covariance_function(identifier):
    """
    Transforms a covariance function identifer to an instance of the
    corresponding covariance function.

    Parameters
    ==========
    identifier : object
        Either an integer, or a list such as [3, 3] where the first
        number is the identifier and the further numbers are parameters
        of the covariance function.

    Returns
    =======
    cov_f : object
        An instance of the specified covariance function.

    Raises
    ------
    ValueError
        Raised when the covariance function identifier is unknown.
    """
    if identifier == 1:
        cov_f = gpr.covariance_functions.RationalQuadraticARD()
    elif identifier == 2:
        cov_f = gpr.covariance_functions.SquaredExponential()
    elif identifier == 3:
        cov_f = gpr.covariance_functions.Matern(5)
    elif isinstance(identifier, list) and identifier[0] == 3:
        cov_f = gpr.covariance_functions.Matern(identifier[1])
    else:
        raise ValueError("Unknown covariance function")

    return cov_f


def _gp_hyp(
    optim_state: dict,
    options: Options,
    plb: np.ndarray,
    pub: np.ndarray,
    gp: gpr.GP,
    X: np.ndarray,
    y: np.ndarray,
    function_logger: FunctionLogger,
):
    """
    Define bounds, priors and samples for GP hyperparameters.

    Parameters
    ==========
    optim_state : dict
        Optimization state from the BADS instance we are calling this from.
    options : Options
        Options from the BADS instance we are calling this from.
    plb : ndarray, shape (hyp_N,)
        Plausible lower bounds for the hyperparameters.
    pub : ndarray, shape (hyp_N,)
        Plausible upper bounds for the hyperparameters.
    gp : GP
        Gaussian process for which we are making the bounds,
        priors and so on.
    X : ndarray, shape (N, D)
        Training inputs.
    y : ndarray, shape (N, 1)
        Training targets.

    Returns
    =======
    gp : GP
        The GP with updates priors, bounds and so on.
    hyp0 : ndarray, shape (hyp_N,)
        Initial guess for the hyperparameters.
    gp_s_N : int
        The number of samples for GP fitting.

    Raises
    ------
    TypeError
        Raised if the mean function is not supported by gpyreg.
    """

    # Get high posterior density dataset.
    hpd_X, hpd_y, _, _ = get_hpd(X, y, options["hpd_frac"])
    D = hpd_X.shape[1]
    # s2 = None

    ## Set GP hyperparameter defaults for BADS.
    cov_bounds_info = gp.covariance.get_bounds_info(
        hpd_X, hpd_y
    )  # in BADS are all zeros
    mean_bounds_info = gp.mean.get_bounds_info(hpd_X, hpd_y)
    noise_bounds_info = gp.noise.get_bounds_info(hpd_X, hpd_y)
    # Missing port: output warping hyperparameters not implemented
    cov_x0 = cov_bounds_info["x0"]
    if isinstance(
        gp.covariance, gpr.gpyreg.covariance_functions.RationalQuadraticARD
    ):
        cov_x0[-1] = 0.0  # shape hyp.

    mean_x0 = mean_bounds_info["x0"]

    noise_x0 = noise_bounds_info["x0"]

    # Hyperparams over observation noise
    if options["fit_lik"]:
        # Unknown noise level (even for deterministic function, it helps for regularization)
        if options.get("specify_target_noise"):
            noise_size = options[
                "tol_fun"
            ]  # Additional jitter to specified noise
        else:
            noise_size = options["noise_size"]

        if np.isscalar(noise_size) or np.size(noise_size) == 1:
            noise_std = 1
            noise_mu = np.log(noise_size)
        else:
            noise_mu = np.log(noise_size[0])
            noise_std = noise_size[1]

    else:
        noise_size = options["tol_fun"]
        noise_mu = np.log(noise_size)

    noise_x0[0] = noise_mu
    hyp0 = np.concatenate([cov_x0, noise_x0, mean_x0])

    # Missing port: output warping hyperparameters not implemented

    ## Change default bounds and set priors over hyperparameters.

    bounds = gp.get_bounds()
    if options["upper_gp_length_factor"] > 0:
        # Max GP input length scale
        bounds["covariance_log_lengthscale"] = (
            -np.inf,
            np.log(options["upper_gp_length_factor"] * (pub - plb)),
        )
    # Increase minimum noise.
    bounds["noise_log_scale"] = (np.log(options["tol_fun"]) - 1, 5)

    # Set priors over hyperparameters
    priors = gp.get_priors()

    tol_mesh = optim_state["tol_mesh"]
    tol_fun = options["tol_fun"]

    D = X.shape[1]
    cov_range = (optim_state["ub"] - optim_state["lb"]) / optim_state["scale"]
    cov_range = (np.minimum(100, 10 * cov_range)).flatten()
    # Bads prior on covariance length scale(s)
    priors["covariance_log_lengthscale"] = ("gaussian", (-1, 2.0))
    # BADS bounds on covariance length scale
    bounds["covariance_log_lengthscale"] = (
        np.array([np.log(tol_mesh)] * D),
        cov_range,
    )  # lower bound and upper bound

    # Bads bounds on signal variance (output scale)
    sf = np.exp(1)
    priors["covariance_log_outputscale"] = ("gaussian", (np.log(sf), 2.0))
    bounds["covariance_log_outputscale"] = (
        np.log(tol_fun),
        np.log(1e6 * tol_fun / tol_mesh),
    )

    if isinstance(
        gp.covariance, gpr.gpyreg.covariance_functions.RationalQuadraticARD
    ):
        rq_prior_mean = 1  # TODO can be assigned by the user
        priors["covariance_log_shape"] = ("gaussian", (rq_prior_mean, 1.0))
        bounds["covariance_log_shape"] = (np.array([-5.0]), np.array([5.0]))

    # Rotate GP axes (unsupported)

    # # Missing port: priors and bounds for output warping hyperparameters
    # (not used)

    # Missing port: we only implement the mean functions that gpyreg supports.
    if isinstance(gp.mean, gpr.mean_functions.ZeroMean):
        pass
    elif isinstance(gp.mean, gpr.mean_functions.ConstantMean):
        # Lower maximum constant mean
        sd = np.std(hpd_y) if len(hpd_y) > 1 else 1.0
        priors["mean_const"] = ("gaussian", (mean_x0, sd))
        bounds["mean_const"] = (mean_bounds_info["LB"], mean_bounds_info["UB"])

    elif isinstance(gp.mean, gpr.mean_functions.NegativeQuadratic):
        if options["gp_quadratic_mean_bound"]:
            delta_y = max(
                options["tol_sd"],
                min(D, np.max(hpd_y) - np.min(hpd_y)),
            )
            bounds["mean_const"] = (-np.inf, np.max(hpd_y) + delta_y)
    else:
        raise TypeError("The mean function is not supported by gpyreg.")

    # Prior over observation noise
    if options["fit_lik"]:
        priors["noise_log_scale"] = ("gaussian", (noise_mu, noise_std))
    else:
        # Known noise level
        priors["noise_log_scale"] = ("delta", (noise_mu))

    gp.temporary_data["len_scale"] = 1
    gp.temporary_data["poll_scale"] = np.ones(D)
    gp.temporary_data["effective_radius"] = 1.0
    # gpstruct.sf = np.exp(1) no need of this since we do not use the hedge acquisition function

    # Missing port: meanfun == 14 hyperprior case

    # Missing port: output warping priors

    ## Number of GP hyperparameter samples.
    gp_s_N = 0

    gp.set_bounds(bounds)
    gp.set_priors(priors)

    return gp, hyp0, round(gp_s_N)


def _get_gp_training_options(
    optim_state: dict,
    iteration_history: IterationHistory,
    options: Options,
    hyp_dict: dict,
    gp_s_N: int,
    function_logger: FunctionLogger,
    second_fit=False,
):
    """
    Get options for training GP hyperparameters.

    Parameters
    ==========
    optim_state : dict
        Optimization state from the BADS instance we are calling this from.
    iteration_history : IterationHistory
        Iteration history from the BADS instance we are calling this from.
    options : Options
        Options from the BADS instance we are calling this from.
    hyp_dict : dict
        Hyperparameter summary statistic dictionary.
    gp_s_N : int
        Number of samples for the GP fitting.

    Returns
    =======
    gp_train : dic
        A dictionary of GP training options.

    Raises
    ------
    ValueError
        Raised if the MCMC sampler for GP hyperparameters is unknown.

    """
    iteration = optim_state["iter"]

    n_eff = np.sum(function_logger.n_evals[function_logger.X_flag])

    gp_train = {}
    gp_train["init_method"] = options["gp_train_init_method"]
    gp_train["tol_opt"] = options["gp_tol_opt"]
    gp_train["widths"] = None

    # Set GP training options
    gp_train["sampler"] = "slicesample"

    # N-dependent initial training points.
    a = -(options["gp_train_n_init"] - options["gp_train_n_init_final"])
    b = -3 * a
    c = 3 * a
    d = options["gp_train_n_init"]
    eff_starting_points = optim_state["eff_starting_points"]
    x = (n_eff - eff_starting_points) / (
        min(options["max_fun_evals"], options["n_train_max"])
        - eff_starting_points
    )
    f = lambda x_: a * x_**3 + b * x**2 + c * x + d
    init_N = max(round(f(x)), options["gp_train_n_init_final"])
    if (
        iteration >= 0
    ):  # the first time is called when the gp is initialized, and iteration is -1
        iteration_history.record("init_N", int(init_N), iteration)
        iteration_history.record(
            "ntrain", int(optim_state["ntrain"]), iteration
        )

    gp_train["init_N"] = init_N
    if second_fit:
        gp_train["opts_N"] = 2
    else:
        gp_train["opts_N"] = 1

    gp_train["n_samples"] = round(gp_s_N)

    return gp_train


def get_grid_search_neighbors(
    function_logger: FunctionLogger, u, gp, options, optim_state
):
    """Retrieve a sorted matrix based on distance from the current incumbent `u`.
    Return
    ----------
        tuple : (U, Y, S)
            U (np.ndarray) : nearest neighbors from the incumbent `u`
            Y (np.ndarray) : predicted values at U
            S (np.ndarray) : estimated variance at U
    """
    # get the training set by retrieving the sorted NEAREST neighbors from u
    U_max_idx = function_logger.X_max_idx
    U = function_logger.X[0 : U_max_idx + 1].copy()
    Y = function_logger.Y[0 : U_max_idx + 1].copy()

    if function_logger.noise_flag:
        S = function_logger.S[0 : U_max_idx + 1]

    dist = udist(
        U,
        u,
        gp.temporary_data["len_scale"],
        optim_state["lb"],
        optim_state["ub"],
        optim_state["scale"],
        optim_state["periodic_vars"],
    )
    if dist.ndim > 1:
        dist = np.min(dist, axis=1)
    sort_idx = np.argsort(dist)  # Ascending sort

    # Keep only points within a certain (rescale) radius from target
    radius = options["gp_radius"] * gp.temporary_data["effective_radius"]
    ntrain = np.minimum(options["n_train_max"], np.sum(dist <= radius**2))

    # Minimum number of point to keep
    ntrain = np.max(
        [
            options["n_train_min"],
            options["n_train_max"] - options["buffer_ntrain"],
            ntrain,
        ]
    )

    # Up to the maximum number of available points
    ntrain = np.minimum(ntrain, function_logger.X_max_idx +1)
    optim_state["ntrain"] = ntrain
    # Take points closest to reference points
    res_S = None
    if function_logger.noise_flag:
        res_S = function_logger.S[sort_idx[0:ntrain]]
    return (U[sort_idx[0:ntrain]], Y[sort_idx[0:ntrain]], res_S)


def _get_fevals_data(function_logger: FunctionLogger):
    """
    Get all evaluated data.

    Parameters
    ==========
    function_logger : FunctionLogger
        Function logger from the BADS instance which we are calling this from.

    Returns
    =======
    x_train, ndarray
        Training inputs.
    y_train, ndarray
        Training targets.
    s2_train, ndarray, optional
        Training data noise variance, if noise is used.
    t_train, ndarray
        Array of the times it took to evaluate the function on the training
        data.
    """

    x = function_logger.X[function_logger.X_flag, :]
    y = function_logger.Y[function_logger.X_flag]
    if function_logger.noise_flag:
        s2 = function_logger.S[function_logger.X_flag] ** 2
    else:
        s2 = None

    # Missing port: noise_shaping

    evals_time = function_logger.fun_eval_time[function_logger.X_flag]

    return x, y, s2, evals_time


def _estimate_noise_(gp: gpr.GP):
    """Estimate GP observation noise at high posterior density.

    Parameters
    ==========
    gp : GP
        The GP for which to perform the estimate.

    Returns
    =======
    est : float
        The estimate of observation noise.
    """

    hpd_top = 0.2
    N, _ = gp.X.shape

    # Subsample high posterior density dataset
    # Sort by descending order, not ascending.
    order = np.argsort(gp.y, axis=None)[::-1]
    hpd_N = math.ceil(hpd_top * N)
    hpd_X = gp.X[order[0:hpd_N]]
    hpd_y = gp.y[order[0:hpd_N]]

    if gp.s2 is not None:
        hpd_s2 = gp.s2[order[0:hpd_N]]
    else:
        hpd_s2 = None

    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    s_N = np.size(gp.posteriors)

    sn2 = np.zeros((hpd_X.shape[0], s_N))

    for s in range(0, s_N):
        hyp = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
        sn2[:, s : s + 1] = gp.noise.compute(hyp, hpd_X, hpd_y, hpd_s2)

    return np.median(np.mean(sn2, axis=1))


def add_and_update_gp(
    function_logger: FunctionLogger,
    gp: gpr.GP,
    x_new,
    y_new,
    sd_new=None,
    options=None,
):
    """
    Quick posterior reupdate of Gaussian process.

    Parameters
    ==========
    gp : GP
        The GP to update.
    function_logger : FunctionLogger
        Function logger from the BADS instance which we are calling this from.
    Returns
    =======
    gp : GP
        The updated Gaussian process.
    """
    gp.X = np.concatenate((gp.X, np.atleast_2d(x_new)))
    gp.y = np.concatenate((gp.y, np.atleast_2d(y_new)))
    if options["specify_target_noise"] and sd_new is not None:
        gp.s2 = np.concatenate((gp.s2, np.atleast_2d(sd_new)))

    gp.update(compute_posterior=True)

    # Missing port: intmean part
    # TODO how is handled the user defined noise
    # TODO should we add a check if the values are well defined?

    return gp
