"""Options and final results of runs on noisy targets, most of them on a
target that returns the standard deviation of its noise
(`specify_target_noise`)."""

import logging

import numpy as np
import pytest

from pybads import BADS

D = 3


def _noisy_sphere(noise_seed):
    """Sphere with Gaussian noise from its own generator."""
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        return float(np.sum(np.atleast_2d(x) ** 2)) + rng.standard_normal()

    return fun


def _noisy_sphere_with_estimated_sd(noise_seed):
    """Sphere with heteroskedastic noise, returning an estimate of its
    standard deviation that varies from call to call, as the estimate of a
    stochastic target does."""
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        y = float(np.sum(np.atleast_2d(x) ** 2))
        sd = 1.0 + 0.1 * np.sqrt(y)
        sd_estimate = sd * np.exp(0.3 * rng.standard_normal())
        return y + sd * rng.standard_normal(), sd_estimate

    return fun


def _make_bads(fun, **options):
    opts = {
        "display": "off",
        "max_fun_evals": 60,
        "random_seed": 7,
        "uncertainty_handling": True,
        "specify_target_noise": True,
    }
    opts.update(options)
    return BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    )


def test_target_noise_turns_on_uncertainty_handling():
    """`specify_target_noise=True` with `uncertainty_handling` left empty
    turns uncertainty handling on, as in MATLAB BADS."""
    bads = _make_bads(
        _noisy_sphere_with_estimated_sd(0), uncertainty_handling=None
    )
    assert bads.options["uncertainty_handling"] is True
    assert bads.optim_state["uncertainty_handling_level"] == 2


def test_target_noise_refuses_uncertainty_handling_off():
    with pytest.raises(ValueError, match="uncertainty_handling"):
        _make_bads(
            _noisy_sphere_with_estimated_sd(0), uncertainty_handling=False
        )


def _warns_noise_size_ignored(caplog):
    return any(
        record.name == "BADS"
        and "options['noise_size'] is ignored" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("noise_size", [1.0, np.array([1.0])], ids=str)
def test_noise_size_ignored_warns(noise_size, caplog):
    with caplog.at_level("WARNING", logger="BADS"):
        _make_bads(_noisy_sphere_with_estimated_sd(0), noise_size=noise_size)
    assert _warns_noise_size_ignored(caplog)


@pytest.mark.parametrize("noise_size", [None, 0.0], ids=str)
def test_noise_size_empty_or_zero_is_silent(noise_size, caplog):
    with caplog.at_level("WARNING", logger="BADS"):
        _make_bads(_noisy_sphere_with_estimated_sd(0), noise_size=noise_size)
    assert not _warns_noise_size_ignored(caplog)


def test_final_estimate_weights_samples_by_precision():
    """`fval` and `fsd` combine the final samples at the returned point by
    the precisions that the target returns, as MATLAB BADS's
    `FinalEstimate` does."""
    result = _make_bads(_noisy_sphere_with_estimated_sd(0)).optimize()
    y = np.asarray(result["yval_vec"], dtype=float)
    sd = np.asarray(result["ysd_vec"], dtype=float)
    assert y.shape == sd.shape == (10,)
    precision = 1 / sd**2
    assert result["fval"] == pytest.approx(
        np.sum(y * precision) / np.sum(precision), rel=1e-12
    )
    assert result["fsd"] == pytest.approx(
        1 / np.sqrt(np.sum(precision)), rel=1e-12
    )
    # The weighting matters: the plain mean of these samples differs.
    assert abs(result["fval"] - np.mean(y)) > 1e-6


def test_final_estimate_from_one_sample():
    """With one final sample, `fval` and `fsd` are the sample and the
    standard deviation that the target returns with it."""
    result = _make_bads(
        _noisy_sphere_with_estimated_sd(0), noise_final_samples=1
    ).optimize()
    y = np.asarray(result["yval_vec"], dtype=float)
    sd = np.asarray(result["ysd_vec"], dtype=float)
    assert y.shape == sd.shape == (1,)
    assert result["fval"] == pytest.approx(y[0], rel=1e-12)
    assert result["fsd"] == pytest.approx(sd[0], rel=1e-12)


def test_final_message_from_one_sample_prints_a_number(caplog):
    """With one final sample and the target's noise SD, the final message
    gives the sample as a number, as MATLAB BADS's does, not an array."""
    with caplog.at_level(logging.INFO, logger="BADS"):
        result = _make_bads(
            _noisy_sphere_with_estimated_sd(0),
            noise_final_samples=1,
            display="iter",
        ).optimize()
    (message,) = [
        record.getMessage()
        for record in caplog.records
        if "Observed function value at minimum" in record.getMessage()
    ]
    assert message.startswith(
        "Observed function value at minimum: "
        f"{float(result['yval_vec'][0])} (1 sample)."
    )


def test_one_final_sample_without_target_noise_adds_the_incumbent():
    """With one final sample and no noise SD from the target, `yval_vec`
    holds the sample and the incumbent's observation, a row of two as in
    MATLAB BADS (the shape of every other `yval_vec`), and `fval` and `fsd`
    are their mean and its standard error."""
    bads = _make_bads(
        _noisy_sphere(0), specify_target_noise=False, noise_final_samples=1
    )
    result = bads.optimize()
    y = result["yval_vec"]
    assert y.shape == (2,)
    assert y[1] == bads.yval
    assert result["fval"] == pytest.approx(np.mean(y), rel=1e-12)
    assert result["fsd"] == pytest.approx(np.std(y, ddof=1) / np.sqrt(2))


@pytest.mark.parametrize(
    "make_fun, target_noise",
    [(_noisy_sphere, False), (_noisy_sphere_with_estimated_sd, True)],
    ids=["inferred_noise", "specified_noise"],
)
def test_noisy_run_in_one_iteration_reports_incumbent(make_fun, target_noise):
    """A noisy run that ends within its first iteration takes no final
    samples: `yval_vec` holds the incumbent's observation and `ysd_vec` is
    None. `max_iter=1` ends the run within its first iteration, before the
    iteration count moves."""
    bads = _make_bads(
        make_fun(0), specify_target_noise=target_noise, max_iter=1
    )
    result = bads.optimize()
    assert np.array_equal(result["yval_vec"], [bads.yval])
    assert result["ysd_vec"] is None


def test_noise_size_zero_with_target_noise_changes_nothing():
    """With `specify_target_noise`, `noise_size` is ignored, as the warning
    says: 0, which the warning proposes, gives the run of an empty
    `noise_size`."""
    empty = _make_bads(_noisy_sphere_with_estimated_sd(0)).optimize()
    zero = _make_bads(
        _noisy_sphere_with_estimated_sd(0), noise_size=0.0
    ).optimize()
    assert np.array_equal(zero["x"], empty["x"])
    assert zero["fval"] == empty["fval"]
    assert zero["func_count"] == empty["func_count"]


@pytest.mark.parametrize("noise_size", [0.0, -1.0, [0.0, 1.0]], ids=str)
def test_noise_size_must_be_positive(noise_size):
    """Without target noise, `noise_size` sets the prior over the noise of
    the GP, from its logarithm; as in MATLAB BADS, it must be positive."""
    with pytest.raises(ValueError, match="noise_size"):
        _make_bads(
            _noisy_sphere(0),
            specify_target_noise=False,
            noise_size=noise_size,
        )


def _warns_noise_size_too_large(caplog):
    return any(
        record.name == "BADS"
        and "the GP cannot represent a noise SD that large"
        in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "noise_size, target_noise, warns",
    [(500.0, False, True), (None, False, False), (500.0, True, False)],
    ids=["large", "default", "large_ignored"],
)
def test_noise_size_above_gp_noise_bound_warns(
    noise_size, target_noise, warns, caplog
):
    """The GP bounds its noise SD at e^5, about 148, as MATLAB BADS does: a
    larger `noise_size` warns that the target should be rescaled, unless
    `specify_target_noise` makes `noise_size` ignored."""
    make_fun = (
        _noisy_sphere_with_estimated_sd if target_noise else _noisy_sphere
    )
    with caplog.at_level("WARNING", logger="BADS"):
        _make_bads(
            make_fun(0),
            specify_target_noise=target_noise,
            noise_size=noise_size,
        )
    assert _warns_noise_size_too_large(caplog) == warns


def test_noise_size_takes_at_most_two_values():
    with pytest.raises(ValueError, match="noise_size"):
        _make_bads(
            _noisy_sphere(0),
            specify_target_noise=False,
            noise_size=[1.0, 1.0, 1.0],
        )


@pytest.mark.parametrize(
    "noise_size, prior_sd",
    [
        ([2.0], 1.0),
        (np.array([2.0]), 1.0),
        ([2.0, 0.5], 0.5),
        ((2.0, np.inf), 1.0),
    ],
    ids=["list", "array", "pair", "pair_without_sd"],
)
def test_noise_size_forms(noise_size, prior_sd):
    """`noise_size` is a scalar or one value, or MATLAB's pair of the base
    noise SD and the SD of the prior over its logarithm (1 when not
    finite). The noise prior of the GP is centred at the log of the base,
    as `mesh_noise_multiplier` is 0 in a noisy run."""
    bads = _make_bads(
        _noisy_sphere(0), specify_target_noise=False, noise_size=noise_size
    )
    result = bads.optimize()
    assert result["fsd"] > 0
    gp = bads.iteration_history.get("gp")[-1]
    prior = gp.get_priors()["noise_log_scale"]
    assert prior[0] == "gaussian"
    assert prior[1][0] == pytest.approx(np.log(2.0))
    assert prior[1][1] == pytest.approx(prior_sd)


def test_final_estimate_without_target_noise():
    """Without target noise, `fval` and `fsd` are the mean of the final
    samples and its standard error, from their standard deviation
    normalized by n - 1, as MATLAB's `std` is."""
    result = _make_bads(
        _noisy_sphere(0), specify_target_noise=False, max_fun_evals=100
    ).optimize()
    y = np.asarray(result["yval_vec"], dtype=float)
    assert y.shape == (10,)
    assert result["fval"] == pytest.approx(np.mean(y), rel=1e-12)
    assert result["fsd"] == pytest.approx(
        np.std(y, ddof=1) / np.sqrt(y.size), rel=1e-12
    )


def test_final_estimate_recorded_at_its_iterate():
    """The final `fval` and `fsd` go into the iteration history at the
    iterate they describe, the returned point. Which iterate a run returns
    depends on its trajectory, so on the machine; of these three seeded
    runs, at least one returns an iterate before the last, where the
    estimate was recorded until 1.1.0."""
    before_last = []
    for seed in range(3):
        bads = _make_bads(
            _noisy_sphere(0),
            specify_target_noise=False,
            max_fun_evals=100,
            random_seed=seed,
        )
        result = bads.optimize()
        history = bads.iteration_history
        fval = history.get("fval").astype(float)
        fsd = history.get("fsd").astype(float)
        (index,) = np.flatnonzero(fval == result["fval"])
        assert fsd[index] == result["fsd"]
        assert np.array_equal(
            np.ravel(history.get("x")[index]), np.ravel(result["x"])
        )
        before_last.append(index < len(fval) - 1)
    assert any(before_last)


def test_inferred_noise_leaves_no_noise_variances_in_the_gp():
    """With uncertainty handling and a target that returns no noise SD, the
    function logger holds no noise SDs and the GP no noise variances of its
    data, as in MATLAB BADS: the GP infers the noise."""
    bads = _make_bads(_noisy_sphere(0), specify_target_noise=False)
    bads.optimize()
    assert not bads.function_logger.noise_flag
    gps = [gp for gp in bads.iteration_history.get("gp") if gp is not None]
    assert len(gps) > 0
    assert all(gp.s2 is None for gp in gps)


def test_iteration_history_keeps_the_gps_as_recorded(monkeypatch):
    """The end-of-iteration re-evaluation of a noisy run leaves the working
    GP in place, as MATLAB BADS does (only the target's hyperparameters move
    to the chosen iterate): no search or poll works on a GP of the iteration
    history, and each stored GP keeps the hyperparameters recorded with it."""
    shared = []
    original_search = BADS._search_step_
    original_poll = BADS._poll_step_

    def check(self, gp):
        slots = self.iteration_history.get("gp")
        if slots is not None:
            shared.append(any(gp is slot for slot in slots))

    def search(self, gp):
        check(self, gp)
        return original_search(self, gp)

    def poll(self, gp):
        check(self, gp)
        return original_poll(self, gp)

    monkeypatch.setattr(BADS, "_search_step_", search)
    monkeypatch.setattr(BADS, "_poll_step_", poll)
    bads = _make_bads(
        _noisy_sphere(0), specify_target_noise=False, max_fun_evals=150
    )
    bads.optimize()
    assert len(shared) > 0 and not any(shared)
    gps = bads.iteration_history.get("gp")
    hyps = bads.iteration_history.get("gp_hyp_full")
    assert len(gps) > 3
    for gp, hyp in zip(gps, hyps):
        assert np.array_equal(gp.get_hyperparameters(as_array=True), hyp)


class _OneSecondTimer:
    """A timer that times every evaluation at 1 s."""

    def start_timer(self, name):
        pass

    def stop_timer(self, name):
        pass

    def get_duration(self, name):
        return 1.0


def test_target_time_counts_every_evaluation_but_the_noise_test(monkeypatch):
    """The target's time, which `overhead` compares with the run's, counts
    the final samples, as MATLAB BADS's does, and leaves out the noise test
    at the starting point, which MATLAB BADS does not time."""
    monkeypatch.setattr(
        "pybads.function_logger.function_logger.Timer", _OneSecondTimer
    )
    bads = _make_bads(
        _noisy_sphere(0), uncertainty_handling=None, specify_target_noise=False
    )
    result = bads.optimize()
    assert bads.optim_state["uncertainty_handling_level"] == 1
    assert result["yval_vec"].shape == (10,)
    assert bads.function_logger.total_fun_eval_time == (
        result["func_count"] - 1
    )


def test_re_estimation_moves_the_incumbent_with_its_value(monkeypatch):
    """When the re-estimation at the end of an iteration finds an earlier
    iterate better by more than `tol_fun`, the incumbent moves to it, its
    location with its value (MATLAB BADS moves the value and leaves `ubest`
    at the old incumbent): `u_best` is the iterate's, and so is the centre
    of the next poll, unless a search moves the incumbent first."""
    moves = []
    pending = {}
    original_re_evaluate = BADS._re_evaluate_history_
    original_search = BADS._search_step_
    original_poll = BADS._poll_step_

    def re_evaluate(self, gp):
        original_re_evaluate(self, gp)
        history = self.iteration_history
        pending["estimates"] = (
            self.optim_state["iter"],
            history.get("fval").astype(float),
            [np.ravel(u) for u in history.get("u")],
        )

    def check(self, step):
        if "estimates" in pending:
            iteration, fval, u = pending.pop("estimates")
            if self.fval != fval[iteration]:
                (index, *_) = np.flatnonzero(fval == self.fval)
                pending["move"] = move = {
                    "u": u[index].copy(),
                    "fval": self.fval,
                    "elsewhere": not np.array_equal(u[index], u[iteration]),
                    "polled": False,
                }
                moves.append(move)
                assert np.array_equal(np.ravel(self.u_best), move["u"])
                assert np.array_equal(
                    np.ravel(self.optim_state["u"]), move["u"]
                )
                assert self.optim_state["fval"] == self.fval
        if step == "poll" and "move" in pending:
            move = pending.pop("move")
            if self.fval == move["fval"]:  # no search has moved it
                move["polled"] = True
                assert np.array_equal(np.ravel(self.u), move["u"])

    def search(self, gp):
        check(self, "search")
        return original_search(self, gp)

    def poll(self, gp):
        check(self, "poll")
        return original_poll(self, gp)

    monkeypatch.setattr(BADS, "_re_evaluate_history_", re_evaluate)
    monkeypatch.setattr(BADS, "_search_step_", search)
    monkeypatch.setattr(BADS, "_poll_step_", poll)
    noise = np.random.default_rng(100)
    bads = BADS(
        lambda x: float(np.sum(x**2) + 0.5 * noise.standard_normal()),
        np.array([1.5, -1.0]),
        np.full(2, -5.0),
        np.full(2, 5.0),
        np.full(2, -2.0),
        np.full(2, 2.0),
        options={
            "display": "off",
            "max_fun_evals": 200,
            "random_seed": 0,
            "uncertainty_handling": True,
        },
    )
    bads.optimize()
    assert any(move["elsewhere"] and move["polled"] for move in moves)
