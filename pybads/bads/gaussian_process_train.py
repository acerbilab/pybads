from asyncio.log import logger
from copy import deepcopy
import math
from statistics import covariance
from turtle import width

import gpyreg as gpr
import numpy as np
from pytest import Function
from gpyreg.slice_sample import SliceSampler

from pybads.function_logger import FunctionLogger
from pybads.search.grid_functions import get_grid_search_neighbors, udist
from scipy.spatial.distance import cdist

from .options import Options
from pybads.stats.get_hpd import get_hpd
from pybads.utils.iteration_history import IterationHistory


def train_gp(
    hyp_dict: dict,
    optim_state: dict,
    function_logger: FunctionLogger,
    iteration_history: IterationHistory,
    options: Options,
    plb: np.ndarray,
    pub: np.ndarray,
):
    """
    Train Gaussian process model.

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
    x_train, y_train, s2_train, t_train = _get_training_data(function_logger)
    D = x_train.shape[1]

    # Heuristic fitness shaping (unused even in MATLAB)
    # if options.FitnessShaping
    #     [y_train,s2_train] = outputwarp_vbmc(X_train,y_train,s2_train,
    #                                           optimState,options);
    #  end

    # Pick the mean function
    mean_f = _meanfun_name_to_mean_function(optim_state["gp_meanfun"])

    # Pick the covariance function.
    covariance_f = _cov_identifier_to_covariance_function(optim_state["gp_covfun"])

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
        optim_state, options, plb, pub, gp, x_train, y_train, function_logger)
    # Initial GP hyperparameters.

    if hyp_dict["hyp"] is None:
        hyp_dict["hyp"] = hyp0.copy()


    # Get GP training options.
    gp_train = _get_gp_training_options(
        optim_state, iteration_history, options, hyp_dict, gp_s_N, function_logger
    )

    # In some cases the model can change so be careful.
    if gp_train["widths"] is not None and np.size(
        gp_train["widths"]
    ) != np.size(hyp0):
        gp_train["widths"] = None

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

    if (
        "hyp_vp" in hyp_dict
        and hyp_dict["hyp_vp"] is not None
        and gp_train["sampler"] == "npv"
    ):
        hyp0 = hyp_dict["hyp_vp"]

    _, _, res = gp.fit(x_train, y_train, s2_train, hyp0=hyp0, options=gp_train)

    if res is not None:
        # Pre-thinning GP hyperparameters
        hyp_dict["full"] = res["samples"]
        hyp_dict["logp"] = res["log_priors"]

        # Missing port: currently not used since we do
        # not support samplers other than slice sampling.
        # if isfield(gpoutput,'hyp_vp')
        #     hypstruct.hyp_vp = gpoutput.hyp_vp;
        # end

        # if isfield(gpoutput,'stepsize')
        #     optimState.gpmala_stepsize = gpoutput.stepsize;
        #     gpoutput.stepsize
        # end

    # TODO: think about the purpose of this line elsewhere in the program.
    # gp.t = t_train

    # Update running average of GP hyperparameter covariance (coarse)
    if hyp_dict["full"] is not None and hyp_dict["full"].shape[1] > 1:
        hyp_cov = np.cov(hyp_dict["full"].T)
        if hyp_dict["run_cov"] is None or options["hyprunweight"] == 0:
            hyp_dict["run_cov"] = hyp_cov
        else:
            w = options["hyprunweight"] ** options["funevalsperiter"]
            hyp_dict["run_cov"] = (1 - w) * hyp_cov + w * hyp_dict["run_cov"]
    else:
        hyp_dict["run_cov"] = None

    # Missing port: sample for GP for debug (not used)

    # Estimate of GP noise around the top high posterior density region
    # We don't modify optim_state to contain sn2hpd here.
    sn2hpd = _estimate_noise_(gp)

    return gp, gp_s_N, sn2hpd, hyp_dict

def local_gp_fitting(gp: gpr.GP, current_point, function_logger:FunctionLogger, options, optim_state, iteration_history: IterationHistory,refit_flag):
    # Local GP approximation on current point
    # Update the GP training set by setting the NEAREST neighbors (Matlab: gpTrainingSet)
    gp.X, gp.y, s2 = get_grid_search_neighbors(function_logger, current_point, gp, options, optim_state)
    D = gp.X.shape[1]
    if s2 is not None:
        gp.s2 = s2                    

    # TODO: Transformation of objective function
    if options['fitnessshaping']:
        #logger.warn("bads:opt:Fitness shaping not implemented yet")
        pass
    
    idx_finite_y = np.isfinite(gp.y)
    if np.any(~idx_finite_y):
        y_idx_penalty = np.argmax(gp.y[idx_finite_y])
        gp.y[~idx_finite_y] = gp.y[y_idx_penalty].copy()
        if 'S' in optim_state:
            gp.s2[~idx_finite_y] = gp.s2[y_idx_penalty]
            
    gp.temporary_data['erry'] = ~idx_finite_y

    #TODO: Rotate dataset (unsupported)

    # Update GP
    
    # Update piors hyperparameters using empirical Bayes method.
    
    if options.get('specifytargetnoise'):    
        noise_size = options['tolfun']   #Additional jitter to specified noise
    else:
        noise_size = options['noisesize']
    
    # Update GP Noise
    gp_priors = gp.get_priors()
    prior_noise = gp_priors['noise_log_scale'] 
    mu_noise_prior = np.log(noise_size) + options['meshnoisemultiplier'] * np.log(optim_state['mesh_size'])
    prior_noise = (prior_noise[0], (mu_noise_prior, prior_noise[1][1]))

    # TODO: warped likelihood (unsupported)

    # Update GP Mean
    y_mean = np.percentile(gp.y, options['gpmeanpercentile'])
    y_range = options['gpmeanrangefun'](y_mean, gp.y)

    prior_mean = None
    if isinstance(gp.mean, gpr.mean_functions.ConstantMean):
        prior_mean = gp_priors['mean_const']
        prior_mean = (prior_mean[0], (y_mean, prior_mean[1][1]))
    
    if prior_mean is not None and ~options['gpfixedmean']:
        prior_mean = (prior_mean[0], (prior_mean[1][0], y_range**(2/4)))
    elif options['gpfixedmean']:
        # TODO: update hyp mean by assigning ymean
        pass 

    # Update GP Covariance length scale
    if options['gpcovprior'] == 'iso':
        dist = udist(gp.X, gp.X, 1, optim_state['lb'], optim_state['ub'], optim_state['scale'], optim_state['periodic_vars'])
        dist = dist.flatten()
        dist = dist[dist != 0]
        if dist.size > 0:
            uu = 0.5 * np.log(np.max(dist))
            ll = 0.5 * np.log(np.min(dist))

            cov_mu = 0.5* (uu + ll)
            cov_sigma = 0.5* (uu - ll)
            
            gp_priors['covariance_log_lengthscale'] = ('gaussian', (cov_mu, cov_sigma))
    
    # TODO Adjust prior length scales for periodic variables (mapped to unit circle)

    # Empirical prior on covariance signal variance ((output scale)
    if options['warpfunc'] == 0:
        sd_y = np.log(np.std(gp.y))
    else:
        # TODO warp function (Matlab  gpdefbads line-code 302)
        pass

    # Re-fit Guassian Process (optimize or sample -- only optimization supported)
    gp_priors['covariance_log_outputscale'] = ('gaussian', (sd_y, 2.**2))
    gp.set_priors(gp_priors)

    old_hyp_gp = gp.get_hyperparameters(as_array=True)
    exit_flag = np.inf
    if refit_flag:
        dic_hyp_gp = gp.get_hyperparameters()
        # In our configuration we have just one sample hyperparameter, in case of multiple we should randomly pick one
        last_dic_hyp_gp = dic_hyp_gp[-1] # Matlab: Initial point #1 is old hyperparameter value (randomly picked)
        # Check for possible high-noise mode
        if np.isscalar(options['noisesize']) or ~np.isfinite(options['noisesize'][1]):
            noise = 1
            is_high_noise = last_dic_hyp_gp['noise_log_scale'] > np.log(options['noisesize']) + 2*noise
        else:
            noise = options['noisesize'][1]
            is_high_noise = last_dic_hyp_gp['noise_log_scale'] > np.log(options['noisesize'][0]) + 2*noise
        
        is_low_mean = False
        if isinstance(gp.mean, gpr.mean_functions.ConstantMean):
            is_low_mean = last_dic_hyp_gp['mean_const'] < np.min(gp.y)
        
        second_fit = options['doublerefit'] | is_high_noise | is_low_mean
        dic_hyp_gp[-1] = last_dic_hyp_gp 
        
        if second_fit:
            # Sample the hyper-params from priors
            old_hyp_gp = gp.get_hyperparameters(as_array=True)
            new_hyp = get_random_sample_from_prior(gp, old_hyp_gp, optim_state, options)
            new_hyp = 0.5 * (new_hyp + old_hyp_gp)
            new_hyp = gp.hyperparameters_to_dict(new_hyp)
            
            if is_high_noise:
                new_hyp[0]["noise_log_scale"] = np.random.randn()-2
            if is_low_mean and isinstance(gp.mean, gpr.mean_functions.ConstantMean):
                new_hyp[0]['mean_const'] = np.median(gp.y)
            
            new_hyp = gp.hyperparameters_from_dict(new_hyp)
            new_hyp = np.minimum(np.maximum(new_hyp, gp.lower_bounds), gp.upper_bounds)
            # TODO check here the optimization when multiple hyper-params are added.
            dic_hyp_gp.append(gp.hyperparameters_to_dict(new_hyp)[-1])
        
        # Matlab: remove undefined points (no need)
        # Matlab uses hyperSVGD when using multiple samples in the hyp. optimization problem.
        
        gp.set_hyperparameters(dic_hyp_gp, compute_posterior=False)
        hyp_gp = gp.get_hyperparameters(as_array=True)
        
        # FIT GP
        gp_s_N = _get_numb_gp_samples(function_logger, optim_state, options)
        gp_train = _get_gp_training_options(optim_state, iteration_history, options, hyp_gp, gp_s_N, function_logger)
        x_train, y_train, s2_train, _ = _get_training_data(function_logger)
        gp, hyp_gp, res, exit_flag = _robust_gp_fit_(gp, x_train, y_train, s2_train, hyp_gp, gp_train, optim_state, options)
        dic_hyp_gp = gp.hyperparameters_to_dict(hyp_gp)

        hyp_n_samples = len(dic_hyp_gp)
        # Update after fitting
        # Gaussian process length scale
        if len(dic_hyp_gp[0]['covariance_log_lengthscale']) > 1:
            len_scale = np.zeros(D)
            for i in range(hyp_n_samples):
                len_scale +=  len_scale + np.exp(dic_hyp_gp[i]['covariance_log_lengthscale'])
            gp.temporary_data['len_scale'] = len_scale
        else:
            gp.temporary_data['len_scale'] = 1.

        # GP-based geometric length scale
        ll = np.zeros((hyp_n_samples, D))
        for i in range(hyp_n_samples):
            ll[i, :] = options['gprescalepoll'] * dic_hyp_gp[i]['covariance_log_lengthscale']
        #ll = exp(sum(bsxfun(@times, gpstruct.hypweight, ll - mean(ll(:))),2))';

        # Take bounded limits
        ub_bounded = optim_state['ub'].copy()
        ub_bounded[~np.isfinite(ub_bounded)] = optim_state['pub'][~np.isfinite(ub_bounded)]
        lb_bounded = optim_state['lb'].copy()
        lb_bounded[~np.isfinite(lb_bounded)] = optim_state['plb'][~np.isfinite(lb_bounded)]

        ll = np.minimum(np.maximum(ll, optim_state['search_mesh_size']), \
                (ub_bounded-lb_bounded)/optim_state['scale']) # Perhaps this should just be PUB - PLB?
        ll = ll - np.mean(ll)
        ll = np.exp(np.sum(ll, axis=0))
        gp.temporary_data['pollscale'] = ll

        if options['useeffectiveradius']:
            if isinstance(gp.covariance, gpr.gpyreg.covariance_functions.RationalQuadraticARD):
                alpha = np.zeros((hyp_n_samples))
                for i in range(hyp_n_samples):
                    alpha[i] = np.exp(dic_hyp_gp[i]['covariance_log_shape'])

                gp.temporary_data['effective_radius'] = np.sqrt(alpha*(np.exp(1/alpha)-1))
    else:
        hyp_gp = old_hyp_gp

    # Matlab defines the signal variability in the GP, but is never used.

    # Recompute posterior
    try:
        gp.update(hyp=hyp_gp)
    except np.linalg.LinAlgError:
        #Posterior update failed (due to Cholesky decomposition)
        gp.set_hyperparameters(old_hyp_gp)
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

def _robust_gp_fit_(gp: gpr.GP, x_train, y_train, s2_train, hyp_gp, gp_train, optim_state, options):
        
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
                new_hyp, _, res = tmp_gp.fit(X, Y, s2, hyp0=new_hyp, options=gp_train)
                break
            except np.linalg.LinAlgError:
                #handle
                success_flag[i_try] = False
                if i_try > options['removepointsaftertries'] -1:
                    idx_drop_out = np.zeros(len(Y)).astype(bool)
                    # Remove closest pair sample
                    dist = cdist(X, X)
                    # Dist is symmetric thus we dont consider the lower triangular and the diagonal of the matrix
                    dist[np.tril_indices(dist.shape[0])] = np.inf
                    
                    # Indices of the minimum elements
                    idx_min = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                    
                    if Y[idx_min[0]] > Y[idx_min[1]]:
                        idx_drop_out[idx_min[0]] = True
                    else:
                        idx_drop_out[idx_min[1]] = True
                    
                    idx_drop_out = np.logical_or(idx_drop_out, (Y > np.percentile(Y, 95)).flatten())
                    X = X[~idx_drop_out]
                    Y = Y[~idx_drop_out]
                    # Remove also user specified noise
                    if tmp_gp.s2 is not None:
                        tmp_gp.s2 = tmp_gp.s2[~idx_drop_out] 
                    
                # Retry with random sample prior
                # if there are multiple hyp samples we take the last one due to the low_mean or high noise.
                sample_hyp_gp = hyp_gp.copy() if len(hyp_gp) == 1 else hyp_gp[-1].copy() 
                new_hyp = get_random_sample_from_prior(tmp_gp, sample_hyp_gp, optim_state, options)
                new_hyp = 0.5 * (new_hyp + sample_hyp_gp)

                nudge = options['noisenudge']
                if nudge is None or len(nudge) == 0:
                    nudge = np.array([0, 0])
                elif len(nudge) == 1:
                    nudge = np.vstack((nudge, 0.5 * nudge[0]))

                # Try increase starting point of noise
                noise_nudge = noise_nudge + nudge[0]

                # Increase gp noise hyp lower bounds
                bounds = tmp_gp.get_bounds()
                noise_bound = bounds["noise_log_scale"]
                noise_bound = (noise_bound[0] + noise_nudge , noise_bound[1])
                bounds["noise_log_scale"] = noise_bound
                tmp_gp.set_bounds(bounds)

                # Try increase starting point of noise
                new_hyp = tmp_gp.hyperparameters_to_dict(new_hyp)
                new_hyp[0]["noise_log_scale"] = new_hyp[0]["noise_log_scale"] + noise_nudge
                new_hyp = tmp_gp.hyperparameters_from_dict(new_hyp)
                tmp_gp.set_hyperparameters(new_hyp, compute_posterior=False)

        if np.any(success_flag):
            # at least one run succeeded
            gp.set_hyperparameters(new_hyp, False)
        if np.any(~success_flag):
            # at least one failed
            if options['gpwarnings']:
                logger.warning(f'bads:gpHyperOptFail: Failed optimization of hyper-parameters ({n_try} attempts). GP approximation might be unreliable.')

        if np.all(~success_flag):
            success = -1
        elif np.all(success_flag):
            success = 1
        else:
            success = 0
        
        return gp, new_hyp, res, success

def get_random_sample_from_prior(gp:gpr.GP, hyp_gp,optim_state, options):
    hyp_sampler_name = options.get("gphypsampler", "slicesample")
    if hyp_sampler_name != 'slicesample':
        raise ValueError("Wrong sampler")

    # Get slice sampler
    sample_f = lambda hyp_: gp._GP__gp_obj_fun(hyp_, False, True)
    
    width = gp.upper_bounds - gp.lower_bounds
    hyp_sampler = SliceSampler(sample_f, hyp_gp.flatten(), width, gp.lower_bounds, gp.upper_bounds, options = {"display": "off", "diagnostics": False})
    new_hyp = hyp_sampler.sample(1, burn=0)['samples'][0]
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
    function_logger: FunctionLogger
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
    hpd_X, hpd_y, _, _ = get_hpd(X, y, options["hpdfrac"])
    D = hpd_X.shape[1]
    # s2 = None

    ## Set GP hyperparameter defaults for BADS.
    cov_bounds_info = gp.covariance.get_bounds_info(hpd_X, hpd_y)
    mean_bounds_info = gp.mean.get_bounds_info(hpd_X, hpd_y)
    noise_bounds_info = gp.noise.get_bounds_info(hpd_X, hpd_y)
    # Missing port: output warping hyperparameters not implemented
    cov_x0 = cov_bounds_info["x0"]
    if isinstance(gp.covariance, gpr.gpyreg.covariance_functions.RationalQuadraticARD):
        cov_x0[-1] = 0.0 # shape hyp. init

    mean_x0 = mean_bounds_info["x0"]

    noise_x0 = noise_bounds_info["x0"]
    
    # Hyperparams over observation noise
    if options['fitlik']:
        # Unknown noise level (even for deterministic function, it helps for regularization)
        if options.get('specifytargetnoise'):    
            noise_size = options['tolfun']   #Additional jitter to specified noise
        else:
            noise_size = options['noisesize']
        
        if np.isscalar(noise_size) or np.size(noise_size) == 1:
            noise_std = 1
            noise_mu = np.log(noise_size)
        else:
            noise_mu = np.log(noise_size[0])
            noise_std = noise_size[1]
        
    else:
        noise_mu = np.log(noise_size[0])

    noise_x0[0] = noise_mu
    hyp0 = np.concatenate([cov_x0, noise_x0, mean_x0])

    # Missing port: output warping hyperparameters not implemented

    ## Change default bounds and set priors over hyperparameters.

    bounds = gp.get_bounds()
    if options["uppergplengthfactor"] > 0:
        # Max GP input length scale
        bounds["covariance_log_lengthscale"] = (
            -np.inf,
            np.log(options["uppergplengthfactor"] * (pub - plb)),
        )
    # Increase minimum noise.
    bounds["noise_log_scale"] = (np.log(options['tolfun']) -1, 5)

    # Set priors over hyperparameters
    priors = gp.get_priors()

    tol_mesh = optim_state['tol_mesh']
    tol_fun = options['tolfun']
    
    D = X.shape[1]
    cov_range = (optim_state['ub'] - optim_state['lb']) / optim_state['scale']   
    cov_range = (np.minimum(100, 10*cov_range)).flatten()
    # Bads prior on covariance length scale(s)
    priors['covariance_log_lengthscale'] = ('gaussian', (-1, 2.**2))
    # BADS bounds on covariance length scale
    bounds['covariance_log_lengthscale'] = (np.array([np.log(tol_mesh)]*D), cov_range) # lower bound and upper bound
    
    # Bads bounds on signal variance (output scale)
    sf = np.exp(1)
    priors['covariance_log_outputscale'] = ('gaussian', (np.log(sf), 2.**2))
    bounds['covariance_log_outputscale'] = (np.log(tol_fun), np.log(1e6 * tol_fun/tol_mesh))

    if isinstance(gp.covariance, gpr.gpyreg.covariance_functions.RationalQuadraticARD):
        rq_prior_mean = 1 #TODO can be assigned by the user
        priors['covariance_log_shape'] = ('gaussian', (rq_prior_mean, 1.**2))
        bounds['covariance_log_shape'] = (np.array([-5.]), np.array([5.]))

    # Rotate GP axes (unsupported)

    # # Missing port: priors and bounds for output warping hyperparameters
    # (not used)

    # Missing port: we only implement the mean functions that gpyreg supports.
    if isinstance(gp.mean, gpr.mean_functions.ZeroMean):
        pass
    elif isinstance(gp.mean, gpr.mean_functions.ConstantMean):
        # Lower maximum constant mean
        bounds["mean_const"] = (-np.inf, np.inf)
        priors["mean_const"] = ('gaussian', (0., 1.**2))
    elif isinstance(gp.mean, gpr.mean_functions.NegativeQuadratic):
        if options["gpquadraticmeanbound"]:
            delta_y = max(
                options["tolsd"],
                min(D, np.max(hpd_y) - np.min(hpd_y)),
            )
            bounds["mean_const"] = (-np.inf, np.max(hpd_y) + delta_y)
    else:
        raise TypeError("The mean function is not supported by gpyreg.")

    # Prior over observation noise
    if options['fitlik']:
        priors["noise_log_scale"] = ("gaussian", (noise_mu, noise_std))
    else:
        # Known noise level
        priors["noise_log_scale"] = ("delta", (noise_mu))

    gp.temporary_data['len_scale'] = 1
    gp.temporary_data['poll_scale'] = np.ones(D)
    gp.temporary_data['effective_radius'] = 1.

    # Missing port: meanfun == 14 hyperprior case

    # Missing port: output warping priors

    ## Number of GP hyperparameter samples.
    gp_s_N = _get_numb_gp_samples(function_logger, optim_state, options)

    gp.set_bounds(bounds)
    gp.set_priors(priors)

    return gp, hyp0, round(gp_s_N)

def _get_numb_gp_samples(function_logger:FunctionLogger, optim_state, options):
    """ 
        Retrieve the number of GP hyperparameter samples.
        
    """
    stop_sampling = optim_state["stop_sampling"]

    tr_N = function_logger.Xn+1 # Number of training inputs

    if stop_sampling == 0 and tr_N != 0:
        # Number of samples
        gp_s_N = options["nsgpmax"] / np.sqrt(tr_N)

        # Maximum sample cutoff
        if optim_state["warmup"]:
            gp_s_N = np.minimum(gp_s_N, options["nsgpmaxwarmup"])
        else:
            gp_s_N = np.minimum(gp_s_N, options["nsgpmaxmain"])

        # Stop sampling after reaching max number of training points
        if tr_N >= options["stablegpsampling"]:
            stop_sampling = tr_N

        # Stop sampling after reaching threshold of variational components
        if optim_state["vpK"] >= options["stablegpvpk"]:
            stop_sampling = tr_N
    else:
        # No training points
        pass

    if stop_sampling > 0:
        gp_s_N = options["stablegpsamples"]

    return gp_s_N


def _get_gp_training_options(
    optim_state: dict,
    iteration_history: IterationHistory,
    options: Options,
    hyp_dict: dict,
    gp_s_N: int,
    function_logger:FunctionLogger
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
    

    n_eff = np.sum(
        function_logger.nevals[function_logger.X_flag]
    )

    gp_train = {}
    gp_train["thin"] = options["gpsamplethin"]  # MCMC thinning
    gp_train["init_method"] = options["gptraininitmethod"]
    gp_train["tol_opt"] = options["gptolopt"]
    gp_train["tol_opt_mcmc"] = options["gptoloptmcmc"]
    gp_train["widths"] = None

    # Get hyperparameter posterior covariance from previous iterations
    hyp_cov = _get_hyp_cov(optim_state, iteration_history, options, hyp_dict)

    if iteration > 0:
        if options["gpsamplewidths"] > 0 and hyp_cov is not None and "rindex" in iteration_history:
            r_index = iteration_history["rindex"][iteration - 1]
    else:
        r_index = np.inf

    # Setup MCMC sampler
    if options["gphypsampler"] == "slicesample":
        gp_train["sampler"] = "slicesample"
        if options["gpsamplewidths"] > 0 and hyp_cov is not None:
            width_mult = np.maximum(options["gpsamplewidths"], r_index)
            hyp_widths = np.sqrt(np.diag(hyp_cov).T)
            gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult

    elif options["gphypsampler"] == "npv":
        gp_train["sampler"] = "npv"

    elif options["gphypsampler"] == "mala":
        gp_train["sampler"] = "mala"
        if hyp_cov is not None:
            gp_train["widths"] = np.sqrt(np.diag(hyp_cov).T)
        if "gpmala_stepsize" in optim_state:
            gp_train["step_size"] = optim_state["gpmala_stepsize"]

    elif options["gphypsampler"] == "slicelite":
        gp_train["sampler"] = "slicelite"
        if options["gpsamplewidths"] > 0 and hyp_cov is not None:
            width_mult = np.maximum(options["gpsamplewidths"], r_index)
            hyp_widths = np.sqrt(np.diag(hyp_cov).T)
            gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult

    elif options["gphypsampler"] == "splitsample":
        gp_train["sampler"] = "splitsample"
        if options["gpsamplewidths"] > 0 and hyp_cov is not None:
            width_mult = np.maximum(options["gpsamplewidths"], r_index)
            hyp_widths = np.sqrt(np.diag(hyp_cov).T)
            gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult

    elif options["gphypsampler"] == "covsample":
        if options["gpsamplewidths"] > 0 and hyp_cov is not None:
            width_mult = np.maximum(options["gpsamplewidths"], r_index)
            if np.all(np.isfinite(width_mult)) and np.all(
                r_index < options["covsamplethresh"]
            ):
                hyp_n = hyp_cov.shape[0]
                gp_train["widths"] = (
                    hyp_cov + 1e-6 * np.eye(hyp_n)
                ) * width_mult ** 2
                gp_train["sampler"] = "covsample"
                gp_train["thin"] *= math.ceil(np.sqrt(hyp_n))
            else:
                hyp_widths = np.sqrt(np.diag(hyp_cov).T)
                gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult
                gp_train["sampler"] = "slicesample"
        else:
            gp_train["sampler"] = "covsample"

    elif options["gphypsampler"] == "laplace":
        if n_eff < 30:
            gp_train["sampler"] = "slicesample"
            if options["gpsamplewidths"] > 0 and hyp_cov is not None:
                width_mult = np.maximum(options["gpsamplewidths"], r_index)
                hyp_widths = np.sqrt(np.diag(hyp_cov).T)
                gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult
        else:
            gp_train["sampler"] = "laplace"

    else:
        raise ValueError("Unknown MCMC sampler for GP hyperparameters")

    # N-dependent initial training points.
    a = -(options["gptrainninit"] - options["gptrainninitfinal"])
    b = -3 * a
    c = 3 * a
    d = options["gptrainninit"]
    x = (n_eff - options["funevalstart"]) / (
        min(options["maxfunevals"], 1e3) - options["funevalstart"]
    )
    f = lambda x_: a * x_ ** 3 + b * x ** 2 + c * x + d
    init_N = max(round(f(x)), 9)

    # Set other hyperparameter fitting parameters
    if "recompute_var_post" in optim_state and optim_state["recompute_var_post"]:
        gp_train["burn"] = gp_train["thin"] * gp_s_N
        gp_train["init_N"] = init_N
        if gp_s_N > 0:
            gp_train["opts_N"] = 1
        else:
            gp_train["opts_N"] = 2
    else:
        gp_train["burn"] = gp_train["thin"] * 3
        if (
            iteration > 1 and "rindex" in iteration_history
            and iteration_history["rindex"][iteration - 1]
            < options["gpretrainthreshold"]
        ):
            gp_train["init_N"] = 0
            if options["gphypsampler"] == "slicelite":
                # TODO: gpretrainthreshold is by default 1, so we get
                #       division by zero. what should the default be?
                gp_train["burn"] = (
                    max(
                        1,
                        math.ceil(
                            gp_train["thin"]
                            * np.log(
                                iteration_history["rindex"][iteration - 1]
                                / np.log(options["gpretrainthreshold"])
                            )
                        ),
                    )
                    * gp_s_N
                )
                gp_train["thin"] = 1
            if gp_s_N > 0:
                gp_train["opts_N"] = 0
            else:
                gp_train["opts_N"] = 1
        else:
            gp_train["init_N"] = init_N
            if gp_s_N > 0:
                gp_train["opts_N"] = 1
            else:
                gp_train["opts_N"] = 2

    gp_train["n_samples"] = round(gp_s_N)
    gp_train["burn"] = round(gp_train["burn"])

    return gp_train


def _get_hyp_cov(
    optim_state: dict,
    iteration_history: IterationHistory,
    options: Options,
    hyp_dict: dict,
):
    """
    Get hyperparameter posterior covariance.

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

    Returns
    =======
    hyp_cov : ndarray, optional
        The hyperparameter posterior covariance if it can be computed.
    """

    if optim_state["iter"] > 0:
        if options["weightedhypcov"]:
            w_list = []
            hyp_list = []
            w = 1
            for i in range(0, optim_state["iter"]):
                if i > 0:
                    # Be careful with off-by-ones compared to MATLAB here
                    diff_mult = 1
                    if 'sKL' in iteration_history:
                        diff_mult = max(
                            1,
                            np.log(
                                iteration_history["sKL"][optim_state["iter"] - i]
                                / options["tolskl"]
                                * options["funevalsperiter"]
                            ),
                        )
                    w *= options["hyprunweight"] ** (
                        options["funevalsperiter"] * diff_mult
                    )
                # Check if weight is getting too small.
                if w < options["tolcovweight"]:
                    break

                hyp = iteration_history["gp_hyp_full"][
                    optim_state["iter"] - 1 - i
                ]
                hyp_n = hyp.shape[1]
                if len(hyp_list) == 0 or np.shape(hyp_list)[2] == hyp.shape[0]:
                    hyp_list.append(hyp.T)
                    w_list.append(w * np.ones((hyp_n, 1)) / hyp_n)

            w_list = np.concatenate(w_list)
            hyp_list = np.concatenate(hyp_list)

            # Normalize weights
            w_list /= np.sum(w_list, axis=0)
            # Weighted mean
            mu_star = np.sum(hyp_list * w_list, axis=0)

            # Weighted covariance matrix
            hyp_n = np.shape(hyp_list)[1]
            hyp_cov = np.zeros((hyp_n, hyp_n))
            for j in range(0, np.shape(hyp_list)[0]):
                hyp_cov += np.dot(
                    w_list[j],
                    np.dot((hyp_list[j] - mu_star).T, hyp_list[j] - mu_star),
                )

            hyp_cov /= 1 - np.sum(w_list ** 2)

            return hyp_cov

        return hyp_dict["run_cov"]

    return None


def _get_training_data(function_logger: FunctionLogger):
    """
    Get training data for building GP surrogate.

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

    x_train = function_logger.X[function_logger.X_flag, :]
    y_train = function_logger.Y[function_logger.X_flag]
    if function_logger.noise_flag:
        s2_train = function_logger.S[function_logger.X_flag] ** 2
    else:
        s2_train = None

    # Missing port: noiseshaping

    t_train = function_logger.fun_evaltime[function_logger.X_flag]

    return x_train, y_train, s2_train, t_train


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


def reupdate_gp(function_logger: FunctionLogger, gp: gpr.GP):
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

    x_train, y_train, s2_train, t_train = _get_training_data(function_logger)
    gp.X = x_train
    gp.y = y_train
    gp.s2 = s2_train
    # Missing port: gp.t = t_train
    gp.update(compute_posterior=True)

    # Missing port: intmean part

    return gp
