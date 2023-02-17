import numpy as np

from pybads.utils.timer import Timer
from pybads.variable_transformer import VariableTransformer


class FunctionLogger:
    """
    Class that evaluates a function and caches its values.

    Parameters
    ----------
    fun : callable
        The function to be logged.
        `fun` must take a vector input and return a scalar value and,
        optionally, the (estimated) SD of the returned value (if the
        function fun is stochastic).
    D : int
        The number of dimensions that the function takes as input.
    noise_flag : bool
        Whether the function fun is stochastic or not.
    uncertainty_handling_level : {0, 1, 2}
        The uncertainty handling level which can be one of
        (0: none; 1: unknown noise level; 2: user-provided noise).
    cache_size : int, optional
        The initial size of caching table (default 500), number of stored function values.
    variable_transformer : VariableTransformer, optional
        A VariableTransformer is required for linear and non-linear transformation of the input space.
        By default None.
    """

    def __init__(
        self,
        fun,
        D: int,
        noise_flag: bool,
        uncertainty_handling_level: int,
        cache_size: int = 500,
        variable_transformer: VariableTransformer = None,
    ):
        self.fun = fun
        self.D: int = D
        self.noise_flag: bool = noise_flag
        self.uncertainty_handling_level: int = uncertainty_handling_level
        self.he_noise_flag = uncertainty_handling_level == 2
        self.transform_variables = variable_transformer is not None
        self.variable_transformer = variable_transformer
        self.cache_size = cache_size

        self.func_count: int = 0
        self.cache_count: int = 0
        self.X_orig = np.full([cache_size, self.D], np.nan)
        self.Y_orig = np.full([cache_size, 1], np.nan)
        self.X = np.full([cache_size, self.D], np.nan)
        self.Y = np.full([cache_size, 1], np.nan)
        self.Y_max = np.nan
        self.n_evals = np.zeros([cache_size, 1])

        if self.noise_flag:
            self.S = np.full([cache_size, 1], np.nan)

        self.Xn: int = -1  # Last filled entry
        self.X_max_idx = -1  # Last filled entry in the cache memory.
        # Use 1D array since this is a boolean mask.
        self.X_flag = np.full((cache_size,), False, dtype=bool)
        self.y_max = float("-Inf")
        self.fun_eval_time = np.full([self.cache_size, 1], np.nan)
        self.total_fun_eval_time = 0.0

        # TODO:  Handle previous evaluations (e.g. from previous run), ref line 51 bads code.

    def __call__(self, x: np.ndarray, record_duplicate_data: bool = True):
        """
        Evaluates the function ``self.fun`` at x and caches values.

        Parameters
        ----------
        x : np.ndarray
            The point at which the function will be evaluated. The shape of x
            should be (1, D) or (D,).
            
        record_duplicate_data : bool, optional (default True)
            Flag to indicate whether the data is added to training data.

        Returns
        -------
        fval : float
            The result of the evaluation.
        SD : float
            The (estimated) SD of the returned value.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if the function value is not a finite real-valued scalar.
        ValueError
            Raise if the (estimated) SD (second function output)
            is not a finite, positive real-valued scalar.
        """

        timer = Timer()
        if x.ndim > 1:
            x = x.squeeze()
        if x.ndim == 0:
            x = np.atleast_1d(x)
        assert x.size == x.shape[0]
        # Convert back to original space
        if self.transform_variables:
            x_orig = self.variable_transformer.inverse_transf(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        wrong_format_target_function = False
        try:
            timer.start_timer("funtime")
            fun_res = self.fun(x_orig)
            timer.stop_timer("funtime")
            if self.he_noise_flag:
                if (type(fun_res) is tuple) and len(fun_res) == 2:
                    fval_orig, fsd = fun_res
                else:
                    wrong_format_target_function = True
                    error_message = (
                        "The `specify_target_noise` option has been set to `True`.\n"
                        + "The target function should return two outputs: the function value and the target noise.\n"
                        + "Please adjust the target function to return two outputs."
                    )
                    raise ValueError(error_message)
            else:
                fval_orig = fun_res
                fsd = None

            if isinstance(fval_orig, np.ndarray):
                # fval_orig can only be an array with size 1 since we support just single evaluation
                fval_orig = fval_orig.item()
            if isinstance(fsd, np.ndarray):
                # fsd can only be an array with size 1 since we support just single evaluation
                fsd = fsd.item()
        except Exception as err:
            if wrong_format_target_function:
                err.args = (error_message,)

            else:
                err.args += (
                    "\n FunctionLogger:FuncError "
                    + "Error in executing the logged function"
                    + "with input: "
                    + str(x_orig),
                )
            raise

        # if fval is an array with only one element, extract that element
        if not np.isscalar(fval_orig) and np.size(fval_orig) == 1:
            fval_orig = np.array(fval_orig).flat[0]

        # Check function value
        if np.any(
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(fval_orig)))

        # Check returned function SD
        if self.he_noise_flag and (
            not np.isfinite(fsd) or not np.isreal(fsd) or fsd <= 0.0
        ):
            error_message = """FunctionLogger:InvalidNoiseValue
                The returned estimated SD (second function output)
                must be a finite, positive real-valued scalar (returned SD:{}"""
            raise ValueError(error_message.format(str(fsd)))

        # record timer stats
        funtime = timer.get_duration("funtime")

        fval, idx = self._record(
            x_orig,
            x,
            fval_orig,
            fsd,
            funtime,
            record_duplicate_data=record_duplicate_data,
        )
        self.func_count += 1

        return fval, fsd, idx

    def add(
        self,
        x: np.ndarray,
        fval_orig: float,
        fsd: float = None,
        fun_eval_time=np.nan,
    ):
        """
        Add a previously evaluated function to the function cache.

        Parameters
        ----------
        x : np.ndarray
            The point at which the function has been evaluated. The shape of x
            should be (1, D) or (D,).
        fval_orig : float
            The result of the evaluation of the function.
        fsd : float, optional
            The (estimated) SD of the returned value (if heteroskedastic noise
            handling is on) of the evaluation of the function, by default None.
        fun_eval_time : float
            The duration of the time it took to evaluate the function,
            by default np.nan.

        Returns
        -------
        fval : float
            The result of the evaluation.
        SD : float
            The (estimated) SD of the returned value.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if the function value is not a finite real-valued scalar.
        ValueError
            Raise if the (estimated) SD (second function output)
            is not a finite, positive real-valued scalar.
        """
        if x.ndim > 1:
            x = x.squeeze()
        if x.ndim == 0:
            x = np.atleast_1d(x)
        assert x.size == x.shape[0]
        # Convert back to original space
        if self.transform_variables:
            x_orig = self.variable_transformer.inverse_transf(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        if self.noise_flag:
            if fsd is None:
                fsd = 1
        else:
            fsd = None

        # Check function value
        if (
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            error_message = """FunctionLogger:InvalidFuncValue:
            The returned function value must be a finite real-valued scalar
            (returned value {})"""
            raise ValueError(error_message.format(str(fval_orig)))

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(fsd) or not np.isfinite(fsd) or not np.isreal(fsd) or fsd <= 0.0
        ):
            error_message = """FunctionLogger:InvalidNoiseValue
                The returned estimated SD (second function output)
                must be a finite, positive real-valued scalar (returned SD:{}"""
            raise ValueError(error_message.format(str(fsd)))

        self.cache_count += 1
        fval, idx = self._record(x_orig, x, fval_orig, fsd, fun_eval_time)
        return fval, fsd, idx

    def finalize(self):
        """
        Remove unused caching entries.
        """
        self.X_orig = self.X_orig[: self.Xn + 1]
        self.Y_orig = self.Y_orig[: self.Xn + 1]

        # in the original matlab version X and Y get deleted
        self.X = self.X[: self.Xn + 1]
        self.Y = self.Y[: self.Xn + 1]

        if self.noise_flag:
            self.S = self.S[: self.Xn + 1]
        self.X_flag = self.X_flag[: self.Xn + 1]
        self.fun_eval_time = self.fun_eval_time[: self.Xn + 1]

    def reset_fun_eval_time(self):
        self.fun_eval_time = np.full([self.cache_size, 1], np.nan)

    def _expand_arrays(self, resize_amount: int = None):
        """
        A private function to extend the rows of the object attribute arrays.

        Parameters
        ----------
        resize_amount : int, optional
            The number of additional rows, by default expand current table
            by 50%.
        """

        if resize_amount is None:
            resize_amount = int(np.max((np.ceil(self.Xn / 2), 1)))

        self.X_orig = np.append(
            self.X_orig, np.full([resize_amount, self.D], np.nan), axis=0
        )
        self.Y_orig = np.append(
            self.Y_orig, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.X = np.append(
            self.X, np.full([resize_amount, self.D], np.nan), axis=0
        )
        self.Y = np.append(self.Y, np.full([resize_amount, 1], np.nan), axis=0)

        if self.noise_flag:
            self.S = np.append(
                self.S, np.full([resize_amount, 1], np.nan), axis=0
            )
        self.X_flag = np.append(
            self.X_flag, np.full((resize_amount,), False, dtype=bool)
        )
        self.fun_eval_time = np.append(
            self.fun_eval_time, np.full([resize_amount, 1], np.nan), axis=0
        )
        self.n_evals = np.append(
            self.n_evals, np.zeros([resize_amount, 1]), axis=0
        )

    def _record(
        self,
        x_orig: float,
        x: float,
        fval_orig: float,
        fsd: float,
        fun_eval_time: float,
        record_duplicate_data: bool = True,
    ):
        """
        A private method to save function values to class attributes.

        Parameters
        ----------
        x_orig : float
            The point at which the function has been evaluated
            (in original space).
        x : float
            The point at which the function has been evaluated
            (in transformed space).
        fval_orig : float
            The result of the evaluation.
        fsd : float
            The (estimated) SD of the returned value
            (if heteroskedastic noise handling is on).
        fun_eval_time : float
            The duration of the time it took to evaluate the function.

        Returns
        -------
        fval : float
            The result of the evaluation.
        idx : int
            The index of the last updated entry.

        Raises
        ------
        ValueError
            Raise if there is more than one match for a duplicate entry.
        """

        # Do not record new data when for example checking the noise of the function at the same point or when building the final estimator (BADS examples).
        if not record_duplicate_data:
            duplicate_flag = np.all(self.X == x, axis=1)
            if np.any(duplicate_flag):
                # Since we do not record the new duplicate point in the training set
                # and we might have more than one duplicate points (e.g for NON-heteroskedastic cases)
                # we register the function evaluation time and increase the counter of function evaluations in the last duplicate
                last_idx = np.argwhere(duplicate_flag)[-1].item()
                N = self.n_evals[last_idx]
                self.fun_eval_time[last_idx] = (
                    N * self.fun_eval_time[last_idx] + fun_eval_time
                ) / (N + 1)
                self.n_evals[last_idx] += 1
                return fval_orig, last_idx
            else:
                return fval_orig, None

        else:
            # check if the noise is heteroskedastic
            if fsd is not None:
                # Like in PyVBMC check if the point has already been evaluated and estimate the noise with new observations
                duplicate_flag = self.X == x
                if np.any(duplicate_flag.all(axis=1)):
                    if np.sum(duplicate_flag.all(axis=1)) > 1:
                        raise ValueError(
                            "More than one match for duplicate entry."
                        )
                    idx = np.argwhere(duplicate_flag)[0, 0]
                    N = self.n_evals[idx]

                    # if fsd is not None: # We already in the case of the heteroskedastic noise
                    tau_n = 1 / self.S[idx] ** 2
                    tau_1 = 1 / fsd**2
                    self.Y[idx] = (tau_n * self.Y[idx] + tau_1 * fval_orig) / (
                        tau_n + tau_1
                    )
                    self.S[idx] = 1 / np.sqrt(tau_n + tau_1)
                    # else:
                    #    self.y_orig[idx] = (N * self.y_orig[idx] + fval_orig) / (N + 1) # We already checked

                    f_val = self.Y[idx]
                    self.Y[idx] = f_val
                    self.fun_eval_time[idx] = (
                        N * self.fun_eval_time[idx] + fun_eval_time
                    ) / (N + 1)
                    self.n_evals[idx] += 1
                    return f_val, idx

            # Add the new point
            self.Xn += 1
            if self.Xn > self.X_orig.shape[0] - 1:
                self._expand_arrays()

            # record function time
            if not np.isnan(fun_eval_time):
                self.fun_eval_time[self.Xn] = fun_eval_time
                self.total_fun_eval_time += fun_eval_time

            self.X_max_idx = np.minimum(self.X_max_idx + 1, self.X.shape[0])
            self.X_orig[self.Xn] = x_orig.copy()
            self.X[self.Xn] = x.copy()
            self.Y_orig[self.Xn] = fval_orig
            fval = fval_orig

            self.Y[self.Xn] = fval
            if fsd is not None:
                self.S[self.Xn] = fsd
            self.X_flag[self.Xn] = True
            self.n_evals[self.Xn] = np.maximum(1, self.n_evals[self.Xn] + 1)
            self.Y_max = np.amax(self.Y[self.X_flag])
            return fval, self.Xn
