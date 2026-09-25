"""Guards on the GP updates that can raise `LinAlgError`.

Three calls of a run can fail with gpyreg's `LinAlgError` ("Singular matrix
for L Cholesky decomposition"): the posterior update after a new point
(`add_and_update_gp`), the rebuild of the local GP (`local_gp_fitting`) and
the prediction of the optimization target (`BADS._get_target_from_gp_`).
A failure leaves a GP whose posteriors match its training data, marks it in
`gp.temporary_data` for a rebuild (`"needs_rebuild"`) and, after a failed
rebuild, for a refit (`"needs_refit"`), and the run carries on. Failures are
injected by wrapping `gpyreg.GP.update` and `GP.set_hyperparameters`, except
in one test, where the Cholesky factorization fails for real."""

import copy
import inspect
import sys
from collections import namedtuple

import gpyreg as gpr
import numpy as np
import pytest

import pybads.bads.bads as bads_module
from pybads import BADS
from pybads.bads.gaussian_process_train import (
    add_and_update_gp,
    local_gp_fitting,
)

D = 3
SITES = ("add_and_update_gp", "local_gp_fitting", "_get_target_from_gp_")
MARKERS = ("needs_rebuild", "needs_refit")

Call = namedtuple("Call", "site method n caller")


@pytest.fixture(autouse=True)
def _restore_global_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _noisy_sphere(noise_seed=0):
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        return _sphere(x) + rng.standard_normal()

    return fun


def _noisy_sphere_with_sd(noise_seed=0):
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        y = _sphere(x)
        sd = 1.0 + 0.1 * np.sqrt(y)
        return y + sd * rng.standard_normal(), sd

    return fun


# Uncertainty level -> target and options. Level 0 predicts the target too,
# since `uncertain_incumbent` is on by default; level 1 with an explicit
# `uncertainty_handling=True` holds NaN noise variances in `gp.s2`.
LEVELS = {
    0: (lambda: _sphere, {}),
    1: (_noisy_sphere, {"uncertainty_handling": True}),
    2: (
        _noisy_sphere_with_sd,
        {"uncertainty_handling": True, "specify_target_noise": True},
    ),
}


def _make_bads(fun, max_fun_evals=60, seed=3, **options):
    opts = {
        "display": "off",
        "max_fun_evals": max_fun_evals,
        "random_seed": seed,
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


def _pybads_frames(frame):
    """Names of the PyBADS functions on the stack from ``frame`` outwards,
    by module name (the gpyreg clone may live under the repository)."""
    names = []
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module.startswith("pybads.") and not module.startswith(
            "pybads.testing"
        ):
            names.append(frame.f_code.co_name)
        frame = frame.f_back
    return names


class Injector:
    """Makes chosen GP updates raise `LinAlgError` before they start.

    Only the outermost gpyreg call counts (`set_hyperparameters` calls
    `update`), only calls that compute a posterior, and only calls whose
    innermost PyBADS frame is one of `SITES`: the updates inside fits and
    the `compute_posterior=False` calls never fail. ``should_fail`` receives
    a `Call` whose ``n`` counts the eligible calls at its site, from 1."""

    def __init__(self, should_fail):
        self.should_fail = should_fail
        self.counts = dict.fromkeys(SITES, 0)
        self.failed = []
        self.depth = 0

    def wrap(self, method):
        original = getattr(gpr.GP, method)
        signature = inspect.signature(original)

        def wrapper(gp, *args, **kwargs):
            if self.depth == 0:
                bound = signature.bind(gp, *args, **kwargs)
                bound.apply_defaults()
                frames = _pybads_frames(sys._getframe(1))
                if (
                    frames
                    and frames[0] in SITES
                    and bound.arguments["compute_posterior"]
                ):
                    site = frames[0]
                    self.counts[site] += 1
                    call = Call(
                        site,
                        method,
                        self.counts[site],
                        frames[1] if len(frames) > 1 else None,
                    )
                    if self.should_fail(call):
                        self.failed.append(call)
                        raise np.linalg.LinAlgError("injected failure")
            self.depth += 1
            try:
                return original(gp, *args, **kwargs)
            finally:
                self.depth -= 1

        return wrapper


@pytest.fixture
def inject(monkeypatch):
    def install(should_fail):
        injector = Injector(should_fail)
        for method in ("update", "set_hyperparameters"):
            monkeypatch.setattr(gpr.GP, method, injector.wrap(method))
        return injector

    return install


def _same(a, b):
    """Equality of nested dicts, tuples, lists and arrays, NaN equal."""
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_same(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if a is None or b is None or isinstance(a, str):
        return a is b or a == b
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _posterior_arrays(gp):
    return [
        {
            name: getattr(post, name, None)
            for name in ("hyp", "alpha", "L", "sW", "L_chol", "sn2_mult")
        }
        for post in gp.posteriors
    ]


def _temporary_data(gp):
    return {
        key: value
        for key, value in gp.temporary_data.items()
        if key not in MARKERS
    }


def _assert_same_gp(gp, reference):
    assert _same(gp.X, reference.X)
    assert _same(gp.y, reference.y)
    assert _same(gp.s2, reference.s2)
    assert _same(gp.get_priors(), reference.get_priors())
    assert _same(
        gp.get_hyperparameters(as_array=True),
        reference.get_hyperparameters(as_array=True),
    )
    assert _same(_posterior_arrays(gp), _posterior_arrays(reference))
    assert _same(_temporary_data(gp), _temporary_data(reference))


def _assert_consistent(gp):
    """The posteriors match the training data: a fresh computation with the
    same hyperparameters predicts the same values."""
    fresh = copy.deepcopy(gp)
    fresh.update(hyp=fresh.get_hyperparameters(as_array=True))
    x = gp.X[:3] + 0.01
    assert all(post.alpha.shape[0] == gp.X.shape[0] for post in gp.posteriors)
    for mine, theirs in zip(gp.predict(x), fresh.predict(x)):
        np.testing.assert_allclose(mine, theirs, rtol=1e-8, atol=1e-10)


Captured = namedtuple("Captured", "bads add local")


def _capture(level):
    """A short run at ``level``, with deep copies of the arguments of the
    3rd `add_and_update_gp` call and of the 2nd `local_gp_fitting` call."""
    make_fun, options = LEVELS[level]
    counts = {"add": 0, "local": 0}
    captured = {}
    original_add = bads_module.add_and_update_gp
    original_local = bads_module.local_gp_fitting

    def spy_add(function_logger, gp, x_new, y_new, sd_new=None, options=None):
        counts["add"] += 1
        if counts["add"] == 3:
            captured["add"] = copy.deepcopy(
                {"gp": gp, "x": x_new, "y": y_new, "sd": sd_new}
            )
        return original_add(function_logger, gp, x_new, y_new, sd_new, options)

    def spy_local(gp, current_point, *args, **kwargs):
        counts["local"] += 1
        if counts["local"] == 2:
            captured["local"] = copy.deepcopy({"gp": gp, "u": current_point})
        return original_local(gp, current_point, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bads_module, "add_and_update_gp", spy_add)
        mp.setattr(bads_module, "local_gp_fitting", spy_local)
        bads = _make_bads(make_fun(), **options)
        bads.optimize()
    return Captured(bads, captured["add"], captured["local"])


@pytest.fixture(scope="module", params=[0, 1, 2], ids=lambda l: f"level{l}")
def captured(request):
    return request.param, _capture(request.param)


# --- add_and_update_gp ---------------------------------------------------


def test_add_without_failure_matches_assign_then_update(captured):
    """Passing the point through `gp.update` recomputes the posteriors as
    the earlier assign-then-update code did, bit for bit."""
    level, c = captured
    gp = copy.deepcopy(c.add["gp"])
    reference = copy.deepcopy(c.add["gp"])
    s2_before = None if gp.s2 is None else gp.s2.copy()
    x, y, sd = c.add["x"], c.add["y"], c.add["sd"]
    out = add_and_update_gp(
        c.bads.function_logger, gp, x, y, sd, c.bads.options
    )
    assert out is gp

    # The code this replaces.
    reference.X = np.concatenate((reference.X, np.atleast_2d(x)))
    reference.y = np.concatenate((reference.y, np.atleast_2d(y)))
    if c.bads.options["specify_target_noise"] and sd is not None:
        reference.s2 = np.concatenate((reference.s2, np.atleast_2d(sd)))
    reference.update(compute_posterior=True)

    assert np.array_equal(gp.X, reference.X)
    assert np.array_equal(gp.y, reference.y)
    assert _same(_posterior_arrays(gp), _posterior_arrays(reference))
    points = np.vstack((gp.X[:4] + 0.01, np.atleast_2d(x)))
    for mine, theirs in zip(gp.predict(points), reference.predict(points)):
        assert np.array_equal(mine, theirs)
    if level == 0:
        assert gp.s2 is None
    elif level == 1:
        # gpyreg gives a point without a noise variance a zero; the noise
        # function does not read these at level 1.
        assert _same(gp.s2, np.vstack((s2_before, [[0.0]])))
    else:
        assert _same(gp.s2, np.vstack((s2_before, np.atleast_2d(sd))))
    assert not gp.temporary_data.get("needs_rebuild", False)


def test_add_failure_leaves_gp_and_marks_it(captured, inject):
    level, c = captured
    injector = inject(lambda call: call.site == "add_and_update_gp")
    gp = copy.deepcopy(c.add["gp"])
    entry = copy.deepcopy(gp)
    posteriors = gp.posteriors
    out = add_and_update_gp(
        c.bads.function_logger,
        gp,
        c.add["x"],
        c.add["y"],
        c.add["sd"],
        c.bads.options,
    )
    assert out is gp
    assert [call[:2] for call in injector.failed] == [
        ("add_and_update_gp", "update")
    ]
    _assert_same_gp(gp, entry)
    assert gp.posteriors is posteriors
    assert gp.temporary_data["needs_rebuild"] is True
    assert not gp.temporary_data.get("needs_refit", False)


def test_add_real_cholesky_failure_restores_gp():
    """A point that duplicates a training input, under a noise variance of
    about exp(-120), makes gpyreg's factorization fail even after its noise
    retries; gpyreg's restore (from 1.3.3) leaves the GP as it was."""
    rng = np.random.default_rng(2)
    gp = gpr.GP(
        D=2,
        covariance=gpr.covariance_functions.RationalQuadraticARD(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = rng.standard_normal((8, 2))
    y = np.sum(X**2, axis=1, keepdims=True)
    hyp = np.array([[3.0, 3.0, 3.0, 3.0, -60.0, 0.0]])
    gp.update(X_new=X, y_new=y, hyp=hyp)
    probe = copy.deepcopy(gp)
    with pytest.raises(np.linalg.LinAlgError):
        probe.update(X_new=X[:1], y_new=y[:1] + 1.0, hyp=hyp)

    entry = copy.deepcopy(gp)
    add_and_update_gp(
        None, gp, X[0], y[0, 0] + 1.0, None, {"specify_target_noise": False}
    )
    _assert_same_gp(gp, entry)
    assert gp.temporary_data["needs_rebuild"] is True


# --- local_gp_fitting ----------------------------------------------------


def _local_fit(c, gp, refit_flag):
    bads = c.bads
    return local_gp_fitting(
        gp,
        c.local["u"],
        bads.function_logger,
        bads.options,
        copy.deepcopy(bads.optim_state),
        bads.iteration_history,
        refit_flag,
        rng=np.random.default_rng(0),
    )


@pytest.mark.parametrize("refit_flag", [False, True], ids=["norefit", "refit"])
def test_local_fit_double_failure_restores_gp(captured, inject, refit_flag):
    level, c = captured
    injector = inject(
        lambda call: call.site == "local_gp_fitting" and call.n <= 2
    )
    gp = copy.deepcopy(c.local["gp"])
    entry = copy.deepcopy(gp)
    out, exit_flag = _local_fit(c, gp, refit_flag)
    assert out is gp
    assert exit_flag == -2
    assert [call[:2] for call in injector.failed] == [
        ("local_gp_fitting", "update"),
        ("local_gp_fitting", "set_hyperparameters"),
    ]
    _assert_same_gp(gp, entry)
    u = np.atleast_2d(c.local["u"])
    for mine, theirs in zip(gp.predict(u), entry.predict(u)):
        assert np.array_equal(mine, theirs)
    assert gp.temporary_data["needs_rebuild"] is True
    assert gp.temporary_data["needs_refit"] is True


def test_local_fit_recovered_failure_is_unchanged(captured, inject):
    """The first update fails and the recovery, the old hyperparameters on
    the new training set, succeeds: exit flag -2 and no marker, as before
    the guards."""
    level, c = captured
    inject(lambda call: call.site == "local_gp_fitting" and call.n == 1)
    gp = copy.deepcopy(c.local["gp"])
    entry = copy.deepcopy(gp)
    gp.temporary_data["needs_rebuild"] = True
    gp.temporary_data["needs_refit"] = True
    out, exit_flag = _local_fit(c, gp, True)
    assert out is gp
    assert exit_flag == -2
    assert _same(
        gp.get_hyperparameters(as_array=True),
        entry.get_hyperparameters(as_array=True),
    )
    _assert_consistent(gp)
    assert not any(gp.temporary_data.get(m, False) for m in MARKERS)


def test_local_fit_success_clears_markers(captured):
    level, c = captured
    gp = copy.deepcopy(c.local["gp"])
    gp.temporary_data["needs_rebuild"] = True
    gp.temporary_data["needs_refit"] = True
    _, exit_flag = _local_fit(c, gp, False)
    assert exit_flag == np.inf
    _assert_consistent(gp)
    assert not any(gp.temporary_data.get(m, False) for m in MARKERS)


# --- _get_target_from_gp_ ------------------------------------------------


def _nan_prediction_for_target(monkeypatch):
    """Makes `GP.predict` return NaN when called by `_get_target_from_gp_`."""
    original = gpr.GP.predict

    def predict(gp, x_star, *args, **kwargs):
        frames = _pybads_frames(sys._getframe(1))
        if frames and frames[0] == "_get_target_from_gp_":
            nan = np.full((np.atleast_2d(x_star).shape[0], 1), np.nan)
            return nan, nan.copy()
        return original(gp, x_star, *args, **kwargs)

    monkeypatch.setattr(gpr.GP, "predict", predict)


def test_target_failure_predicts_from_current_gp(captured, inject):
    """When the posterior under the best iteration's hyperparameters cannot
    be computed, the target is predicted from the GP as it stands."""
    level, c = captured
    bads = c.bads
    gp = copy.deepcopy(c.add["gp"])
    hyp_best = gp.get_hyperparameters(as_array=True) + 0.1
    u = np.atleast_2d(bads.u_best)
    injector = inject(lambda call: call.site == "_get_target_from_gp_")
    f_target_mu, f_target_s, f_target = bads._get_target_from_gp_(
        bads.u_best, gp, hyp_best
    )
    assert [call[:2] for call in injector.failed] == [
        ("_get_target_from_gp_", "set_hyperparameters")
    ]
    mu, s2 = gp.predict(u)
    # The call sites store these with `.item()`.
    assert f_target_mu.item() == mu.item()
    assert np.isfinite(f_target.item())
    assert np.asarray(f_target_s).item() == pytest.approx(np.sqrt(s2.item()))


@pytest.mark.parametrize(
    "fail_hyperparameters", [False, True], ids=["nan", "failure_then_nan"]
)
def test_target_fallback_to_incumbent(
    captured, inject, monkeypatch, fail_hyperparameters
):
    """A prediction that is not finite falls back to the incumbent's `fval`
    and `fsd`, in a form the call sites' `.item()` accepts."""
    level, c = captured
    bads = c.bads
    gp = copy.deepcopy(c.add["gp"])
    if fail_hyperparameters:
        inject(lambda call: call.site == "_get_target_from_gp_")
    _nan_prediction_for_target(monkeypatch)
    f_target_mu, f_target_s, f_target = bads._get_target_from_gp_(
        bads.u_best, gp, gp.get_hyperparameters(as_array=True)
    )
    assert f_target_mu.item() == bads.optim_state["fval"]
    assert f_target_s == bads.optim_state["fsd"]
    f_target.item()


# --- the markers in the search and the poll --------------------------------


def _probe_search(monkeypatch, markers):
    """On the first search step that would not rebuild the GP by itself
    (`search_count > 0`, `reset_gp` false, no refit due), sets ``markers``
    on its GP; returns the `refit_flag` of each `local_gp_fitting` call
    made during that step."""
    original_step = BADS._search_step_
    original_local = bads_module.local_gp_fitting
    state = {"probing": False, "done": False, "calls": []}

    def spy_local(
        gp,
        current_point,
        function_logger,
        options,
        optim_state,
        iteration_history,
        refit_flag,
        rng=None,
    ):
        if state["probing"]:
            state["calls"].append(refit_flag)
        return original_local(
            gp,
            current_point,
            function_logger,
            options,
            optim_state,
            iteration_history,
            refit_flag,
            rng=rng,
        )

    def step(self, gp):
        if state["done"] or self.optim_state["search_count"] == 0:
            return original_step(self, gp)
        state["probing"] = state["done"] = True
        self.reset_gp = False
        self._is_gp_refit_time_ = lambda alpha: (False, False)
        for marker in markers:
            gp.temporary_data[marker] = True
        try:
            return original_step(self, gp)
        finally:
            del self._is_gp_refit_time_
            state["probing"] = False

    monkeypatch.setattr(bads_module, "local_gp_fitting", spy_local)
    monkeypatch.setattr(BADS, "_search_step_", step)
    _make_bads(_sphere, max_fun_evals=80).optimize()
    assert state["done"]
    return state["calls"]


def _probe_poll(monkeypatch, markers):
    """In the first poll step, sets ``markers`` on the GP right after the
    rebuild of its first iteration, with `reset_gp` false and no refit due;
    returns the `refit_flag` of each `local_gp_fitting` call of that poll."""
    original_step = BADS._poll_step_
    original_local = bads_module.local_gp_fitting
    state = {"probing": False, "done": False, "calls": []}

    def spy_local(
        gp,
        current_point,
        function_logger,
        options,
        optim_state,
        iteration_history,
        refit_flag,
        rng=None,
    ):
        out = original_local(
            gp,
            current_point,
            function_logger,
            options,
            optim_state,
            iteration_history,
            refit_flag,
            rng=rng,
        )
        if state["probing"]:
            if not state["calls"]:
                for marker in markers:
                    out[0].temporary_data[marker] = True
            state["calls"].append(refit_flag)
        return out

    def step(self, gp):
        if state["done"]:
            return original_step(self, gp)
        state["probing"] = state["done"] = True
        self.reset_gp = False
        self._is_gp_refit_time_ = lambda alpha: (False, False)
        try:
            return original_step(self, gp)
        finally:
            del self._is_gp_refit_time_
            state["probing"] = False

    monkeypatch.setattr(bads_module, "local_gp_fitting", spy_local)
    monkeypatch.setattr(BADS, "_poll_step_", step)
    # A complete poll evaluates every direction, so the poll has several
    # iterations after its first rebuild.
    _make_bads(_sphere, max_fun_evals=80, complete_poll=True).optimize()
    assert state["done"]
    return state["calls"]


def test_search_rebuilds_marked_gp(monkeypatch):
    assert _probe_search(monkeypatch, ["needs_rebuild"]) == [False]


def test_search_leaves_unmarked_gp(monkeypatch):
    assert _probe_search(monkeypatch, []) == []


def test_search_refits_after_failed_rebuild(monkeypatch):
    assert _probe_search(monkeypatch, MARKERS) == [True]


def test_poll_rebuilds_marked_gp(monkeypatch):
    # The first rebuild, and one more for the marker, which it clears.
    assert _probe_poll(monkeypatch, ["needs_rebuild"]) == [False, False]


def test_poll_leaves_unmarked_gp(monkeypatch):
    assert _probe_poll(monkeypatch, []) == [False]


def test_poll_refits_after_failed_rebuild(monkeypatch):
    assert _probe_poll(monkeypatch, MARKERS) == [False, True]


# --- noisy poll after a failed add ----------------------------------------


def test_noisy_poll_after_failed_add_counts_no_improvement(
    inject, monkeypatch
):
    """The GP does not hold the point, so its estimate there is NaN, as in
    MATLAB BADS, and the point counts as no improvement."""
    injector = inject(
        lambda call: call.site == "add_and_update_gp"
        and call.caller == "_poll_step_"
        and not injector.failed
    )
    original = BADS._eval_improvement_
    seen = []

    def spy(self, f_base, f_new, s_base, s_new, q):
        if injector.failed and not seen:
            seen.append((f_new, s_new))
        return original(self, f_base, f_new, s_base, s_new, q)

    monkeypatch.setattr(BADS, "_eval_improvement_", spy)
    make_fun, options = LEVELS[1]
    result = _make_bads(make_fun(), max_fun_evals=100, **options).optimize()
    assert injector.failed
    assert len(seen) == 1 and np.isnan(seen[0][0]) and np.isnan(seen[0][1])
    assert np.isfinite(result["fval"])


# --- whole runs ------------------------------------------------------------


@pytest.mark.parametrize(
    "level, should_fail",
    [
        (0, lambda call: call.site == "add_and_update_gp" and call.n % 5 == 0),
        (2, lambda call: call.site == "add_and_update_gp" and call.n % 5 == 0),
        (1, lambda call: call.site == "local_gp_fitting" and call.n in (5, 6)),
        (1, lambda call: call.site == "_get_target_from_gp_" and call.n == 3),
    ],
    ids=["add_level0", "add_level2", "double_local_level1", "target_level1"],
)
def test_run_carries_on_after_failures(inject, level, should_fail):
    injector = inject(should_fail)
    make_fun, options = LEVELS[level]
    result = _make_bads(make_fun(), max_fun_evals=150, **options).optimize()
    assert injector.failed
    assert np.isfinite(result["fval"])
    assert np.all(np.isfinite(result["x"]))
