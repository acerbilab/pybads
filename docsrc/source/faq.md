# PyBADS: Frequently Asked Questions

This FAQ is curated by [Luigi Acerbi](https://lacerbi.github.io/), and in constant expansion.
It is adapted for PyBADS 1.5 from the [MATLAB BADS FAQ](https://github.com/acerbilab/bads/wiki),
with further questions on the Python package.

For a tutorial with detailed examples, see the [Jupyter notebook examples](examples.rst).

If you have questions not covered here, please feel free to ask in the lab
[Discussions forum](https://github.com/orgs/acerbilab/discussions).

The snippets below use `np` for NumPy and `BADS` for the optimizer class:

```python
import numpy as np
from pybads import BADS
```

Supply your objective function `fun`, starting point `x0`, and bounds `lb`,
`ub`, `plb`, `pub` where they appear in a snippet. The bounds are the
arguments of `BADS`:

| Variable | `BADS` argument |
| --- | --- |
| `lb` | `lower_bounds` |
| `ub` | `upper_bounds` |
| `plb` | `plausible_lower_bounds` |
| `pub` | `plausible_upper_bounds` |

Give the starting point and each bound as a one-dimensional NumPy array of
`D` elements, one per variable. A list, or an array of shape `(1, D)`, is
taken too. A scalar bound applies to every variable when `x0` is given;
with `x0=None`, PyBADS counts the variables from the plausible bounds (or,
without them, from the hard bounds), so give those as arrays.

## Table of contents

- [General](#faq-general)
  - [Which kind of problems is PyBADS suited for?](#faq-which-kind-of-problems-is-pybads-suited-for)
  - [The performance of BADS on the real model-fitting problems reported in the paper is remarkable. Did you cherry-pick the results?](#faq-the-performance-of-bads-on-the-real-model-fitting-problems-reported-in-the-paper-is-remarkable-did-you-cherry-pick-the-results)
  - [What do I do if PyBADS is not suited for my problem?](#faq-what-do-i-do-if-pybads-is-not-suited-for-my-problem)
- [Installing PyBADS](#faq-installation)
  - [Where can I download PyBADS?](#faq-where-can-i-download-pybads)
  - [Which external packages does PyBADS require?](#faq-which-external-packages-does-pybads-require)
  - [Which version of Python do I need?](#faq-which-version-of-python-do-i-need)
  - [I am having trouble installing PyBADS. Can you help?](#faq-i-am-having-trouble-installing-pybads-can-you-help)
- [Input arguments (objective function: `fun`)](#faq-input-arguments-objective-function-fun)
  - [What is the objective function?](#faq-what-is-the-objective-function)
  - [Why the *negative* log likelihood?](#faq-why-the-negative-log-likelihood)
  - [My objective function requires additional data/inputs. How do I pass them to PyBADS?](#faq-my-objective-function-requires-additional-datainputs-how-do-i-pass-them-to-pybads)
- [Input arguments (domain: `x0`, `lb`, `ub`, `plb`, `pub`, `non_box_cons`)](#faq-input-arguments-domain-x0-lb-ub-plb-pub-non_box_cons)
  - [How do I choose the starting point `x0`?](#faq-how-do-i-choose-the-starting-point-x0)
  - [How do I run PyBADS from several starting points?](#faq-how-do-i-run-pybads-from-several-starting-points)
  - [How do I choose `lb` and `ub`?](#faq-how-do-i-choose-lb-and-ub)
  - [What if I really have no idea how to choose `lb` and `ub`?](#faq-what-if-i-really-have-no-idea-how-to-choose-lb-and-ub)
  - [Does PyBADS support (partially) unconstrained optimization?](#faq-does-pybads-support-partially-unconstrained-optimization)
  - [Can I set `lb = ub` for some variable to fix it to a given value?](#faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value)
  - [How do I choose `plb` and `pub`?](#faq-how-do-i-choose-plb-and-pub)
  - [Does PyBADS rescale or transform my variables?](#faq-does-pybads-rescale-or-transform-my-variables)
  - [How do I prevent PyBADS from evaluating certain inputs or regions of input space?](#faq-how-do-i-prevent-pybads-from-evaluating-certain-inputs-or-regions-of-input-space)
  - [Does PyBADS support integer constraints?](#faq-does-pybads-support-integer-constraints)
  - [Does PyBADS support periodic variables, such as angles?](#faq-does-pybads-support-periodic-variables-such-as-angles)
- [Output arguments](#faq-output-arguments)
  - [What does `optimize()` return?](#faq-what-does-optimize-return)
  - [How is `fval` computed?](#faq-how-is-fval-computed)
  - [Why do you estimate `fval` by averaging additional function evaluations? Can't you return the Gaussian process prediction at `x`?](#faq-why-do-you-estimate-fval-by-averaging-additional-function-evaluations-cant-you-return-the-gaussian-process-prediction-at-x)
  - [How is `overhead` defined?](#faq-how-is-overhead-defined)
  - [Where can I find the internal state and iteration history?](#faq-where-can-i-find-the-internal-state-and-iteration-history)
  - [Is it possible to output or inspect the optimization trajectory?](#faq-is-it-possible-to-output-or-inspect-the-optimization-trajectory)
  - [Can I monitor or stop a run while it is running?](#faq-can-i-monitor-or-stop-a-run-while-it-is-running)
- [Noisy objective function](#faq-noisy-objective-function)
  - [What is a *noisy* objective function?](#faq-what-is-a-noisy-objective-function)
  - [Why are noisy objective functions treated differently?](#faq-why-are-noisy-objective-functions-treated-differently)
  - [Can I make a noisy objective function deterministic by fixing the noise process?](#faq-can-i-make-a-noisy-objective-function-deterministic-by-fixing-the-noise-process)
  - [Should I tell PyBADS that my objective is noisy?](#faq-should-i-tell-pybads-that-my-objective-is-noisy)
  - [Can PyBADS handle any arbitrary amount of noise in the objective?](#faq-can-pybads-handle-any-arbitrary-amount-of-noise-in-the-objective)
  - [Does PyBADS assume that the noise is the same for all inputs?](#faq-does-pybads-assume-that-the-noise-is-the-same-for-all-inputs)
  - [Should I provide an estimate of the noise associated with each evaluation?](#faq-should-i-provide-an-estimate-of-the-noise-associated-with-each-evaluation)
  - [How do I estimate the standard deviation of a noisy objective?](#faq-how-do-i-estimate-the-standard-deviation-of-a-noisy-objective)
- [Display](#faq-display)
  - [What are the quantities displayed by PyBADS during optimization?](#faq-what-are-the-quantities-displayed-by-pybads-during-optimization)
  - [For a noisy function, I noticed that the series of displayed `E[f(x)]` values is *not* monotonically decreasing. Should I worry?](#faq-for-a-noisy-function-i-noticed-that-the-series-of-displayed-efx-values-is-not-monotonically-decreasing-should-i-worry)
  - [For a noisy function, I noticed that the series of displayed `SD[f(x)]` sometimes shows sudden jumps (e.g., from ~1 to ~4). Is that normal?](#faq-for-a-noisy-function-i-noticed-that-the-series-of-displayed-sdfx-sometimes-shows-sudden-jumps-eg-from-1-to-4-is-that-normal)
  - [Sometimes as `Actions` during optimization I read `Train (failed)`. What does it mean?](#faq-sometimes-as-actions-during-optimization-i-read-train-failed-what-does-it-mean)
  - [How do I silence PyBADS, or send its output elsewhere?](#faq-how-do-i-silence-pybads-or-send-its-output-elsewhere)
- [Troubleshooting](#faq-troubleshooting)
  - [Is there a way to check that PyBADS is running correctly — is it enough that it does not give warnings/errors?](#faq-is-there-a-way-to-check-that-pybads-is-running-correctly-is-it-enough-that-it-does-not-give-warningserrors)
  - [PyBADS crashes saying that `The returned function value must be a finite real-valued scalar`. What do I do?](#faq-pybads-crashes-saying-that-the-returned-function-value-must-be-a-finite-real-valued-scalar-what-do-i-do)
  - [During optimization I received a warning that `The mesh attempted to expand above maximum size too many times`. What does it mean?](#faq-during-optimization-i-received-a-warning-that-the-mesh-attempted-to-expand-above-maximum-size-too-many-times-what-does-it-mean)
  - [I am passing `non_box_cons` to PyBADS, but I get an error that `non_box_cons should be a function that takes an N x D array X`. What am I doing wrong?](#faq-i-am-passing-non_box_cons-to-pybads-but-i-get-an-error-that-non_box_cons-should-be-a-function-that-takes-an-n-x-d-array-x-what-am-i-doing-wrong)
  - [I have been running PyBADS with a *deterministic* objective function from the *same* starting point, but I get different results each time. Is something wrong?](#faq-i-have-been-running-pybads-with-a-deterministic-objective-function-from-the-same-starting-point-but-i-get-different-results-each-time-is-something-wrong)
  - [How do I make a run reproducible?](#faq-how-do-i-make-a-run-reproducible)
  - [I have been running PyBADS with a *stochastic* objective function from different starting points and I get different results each time. What can I do?](#faq-i-have-been-running-pybads-with-a-stochastic-objective-function-from-different-starting-points-and-i-get-different-results-each-time-what-can-i-do)
  - [On some problems, PyBADS seems to get stuck and stop too early. Is there a way to tune PyBADS to optimize towards a higher precision result or to have it optimize for longer?](#faq-on-some-problems-pybads-seems-to-get-stuck-and-stop-too-early-is-there-a-way-to-tune-pybads-to-optimize-towards-a-higher-precision-result-or-to-have-it-optimize-for-longer)
  - [On some problems, PyBADS seems to find a reasonably good solution, but then it takes a long time to converge, spending many iterations at very small values of `MeshScale`. Is there a way to tune PyBADS to stop earlier once it finds a decent solution?](#faq-on-some-problems-pybads-seems-to-find-a-reasonably-good-solution-but-then-it-takes-a-long-time-to-converge-spending-many-iterations-at-very-small-values-of-meshscale-is-there-a-way-to-tune-pybads-to-stop-earlier-once-it-finds-a-decent-solution)
- [Miscellanea](#faq-miscellanea)
  - [This is interesting, but shouldn't we ideally compute full posterior distributions?](#faq-this-is-interesting-but-shouldnt-we-ideally-compute-full-posterior-distributions)
  - [Can PyBADS return an approximate posterior, e.g. by computing the Hessian at the optimum?](#faq-can-pybads-return-an-approximate-posterior-eg-by-computing-the-hessian-at-the-optimum)
  - [I have run PyBADS on my problem. How do I run PyVBMC?](#faq-i-have-run-pybads-on-my-problem-how-do-i-run-pyvbmc)
  - [I used BADS in MATLAB. What is different in PyBADS?](#faq-i-used-bads-in-matlab-what-is-different-in-pybads)
  - [Are you planning to port BADS to other languages?](#faq-are-you-planning-to-port-bads-to-other-languages)

(faq-general)=
## General

(faq-which-kind-of-problems-is-pybads-suited-for)=
### Which kind of problems is PyBADS suited for?

We recommend PyBADS for problems in which:

<!-- index.rst includes this list, between the two markers suited-for, which it matches exactly: a changed marker drops the list from the index without failing the build. README.md copies the list. -->
<!-- suited-for: start -->
- the objective function landscape is *rough* (nonsmooth), typically due to numerical approximations or noise;
- the objective function is at least moderately expensive to compute (e.g., more than 0.1 s per function evaluation);
- the gradient is unavailable;
- the number of input parameters is up to about `D = 20`.
<!-- suited-for: end -->

If your objective function is fully analytical, PyBADS is most likely not suited for your problem (see [below](#faq-what-do-i-do-if-pybads-is-not-suited-for-my-problem)).

(faq-the-performance-of-bads-on-the-real-model-fitting-problems-reported-in-the-paper-is-remarkable-did-you-cherry-pick-the-results)=
### The performance of BADS on the real model-fitting problems reported in the paper is remarkable. Did you cherry-pick the results?

No, but we selected projects that we thought BADS would have been [suitable for](#faq-which-kind-of-problems-is-pybads-suited-for).
The benchmark of the [paper](https://arxiv.org/abs/1705.04405) ran the MATLAB implementation of BADS, and some of its problems have been replicated with PyBADS.

(faq-what-do-i-do-if-pybads-is-not-suited-for-my-problem)=
### What do I do if PyBADS is not suited for my problem?

If the objective function is smooth and analytical, we would recommend a
gradient-based optimizer instead, such as those of
[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
(possibly feeding it the analytically calculated gradient).
If you can afford tens or even hundreds of thousands of function evaluations,
CMA-ES with *active* covariance adaptation, as implemented in the
[`cma`](https://github.com/CMA-ES/pycma) package, is also a valid alternative.

In these cases, you may also consider [computing the full posterior](#faq-this-is-interesting-but-shouldnt-we-ideally-compute-full-posterior-distributions),
instead of getting only a point estimate via optimization. To this end, you
could use Markov Chain Monte Carlo, e.g. via [Stan](https://mc-stan.org/) or
[PyMC](https://www.pymc.io/). Alternatively, we developed a method to compute
approximate posterior distributions,
[Variational Bayesian Monte Carlo (PyVBMC)](https://acerbilab.github.io/pyvbmc/),
which can be used in synergy with PyBADS.

(faq-installation)=
## Installing PyBADS

(faq-where-can-i-download-pybads)=
### Where can I download PyBADS?

Install PyBADS with:

```console
python -m pip install pybads
```

or with Conda:

```console
conda install --channel=conda-forge pybads
```

See the [installation instructions](installation.rst) for more details, and
the [GitHub repository](https://github.com/acerbilab/pybads) for the source code.

(faq-which-external-packages-does-pybads-require)=
### Which external packages does PyBADS require?

PyBADS runs on NumPy, SciPy and matplotlib, and builds its Gaussian process
models with [gpyreg](https://github.com/acerbilab/gpyreg), the Gaussian
process library of our lab. `pip` and `conda` install them together with
PyBADS. PyBADS does not require MATLAB.

To run the [example notebooks](examples.rst) you also need Jupyter (see the
[installation instructions](installation.rst)).

(faq-which-version-of-python-do-i-need)=
### Which version of Python do I need?

PyBADS requires Python 3.10 or newer.

(faq-i-am-having-trouble-installing-pybads-can-you-help)=
### I am having trouble installing PyBADS. Can you help?

Sure. The PyBADS installation should be pretty straightforward, so tell us
in detail which problem you are having in the
[Discussions forum](https://github.com/orgs/acerbilab/discussions). Include
your operating system, Python version, installation command and the full
error message.

(faq-input-arguments-objective-function-fun)=
## Input arguments (objective function: `fun`)

(faq-what-is-the-objective-function)=
### What is the objective function?

The *objective* or *target* function is the function that you want PyBADS to
minimize, a Python callable `fun` passed as the first argument of `BADS`.

For a typical model-fitting problem, `fun` is a function that computes the
negative log likelihood of an input parameter vector `x`, for a given dataset
and model.

`fun` takes a one-dimensional NumPy array of shape `(D,)`, a single point,
and returns a finite real scalar: PyBADS evaluates one point per call. For
example, to fit the mean and the log standard deviation of normally
distributed data:

```python
from scipy.stats import norm

data = np.random.default_rng(0).normal(1.0, 2.0, size=100)


def fun(x):
    mu, log_sigma = x  # x has shape (2,)
    return -np.sum(norm.logpdf(data, loc=mu, scale=np.exp(log_sigma)))


bads = BADS(fun, x0, lb, ub, plb, pub)
optimize_result = bads.optimize()
x_min = optimize_result["x"]
fval = optimize_result["fval"]
```

A [noisy](#faq-noisy-objective-function) objective that can estimate its own
noise returns a pair instead, as explained
[below](#faq-should-i-provide-an-estimate-of-the-noise-associated-with-each-evaluation).

(faq-why-the-negative-log-likelihood)=
### Why the *negative* log likelihood?

By mathematical convention, PyBADS *minimizes* the objective function, as most
other optimization algorithms.

In the typical model-fitting scenario we want to
[maximize the likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).
Which is the same as maximizing the log likelihood. Which is the same as
minimizing minus the log likelihood, *aka* the negative log likelihood.

More generally, to maximize a function `g`, minimize `-g`, and flip the sign
of the returned `fval`:

```python
bads = BADS(lambda x: -g(x), x0, lb, ub, plb, pub)
optimize_result = bads.optimize()
g_max = -optimize_result["fval"]
```

(faq-my-objective-function-requires-additional-datainputs-how-do-i-pass-them-to-pybads)=
### My objective function requires additional data/inputs. How do I pass them to PyBADS?

Suppose that your function takes two inputs, `fun(x, data)`.

The first solution consists of defining a new function

```python
def funwdata(x):
    return fun(x, data)
```

where `data` has been defined before in the code. Now you can optimize
`funwdata`, which takes a single input. The new function looks up `data`
each time it is called, so keep the data fixed while an optimization runs:
a change to it changes the objective.

Alternatively, you can use `functools.partial`:

```python
from functools import partial

funwdata = partial(fun, data=data)
bads = BADS(funwdata, x0, lb, ub, plb, pub)
```

which binds the `data` argument to the object passed.

`BADS` takes no extra arguments to pass on to the objective, unlike the
MATLAB `bads` function, which passes those that follow `OPTIONS`.

(faq-input-arguments-domain-x0-lb-ub-plb-pub-non_box_cons)=
## Input arguments (domain: `x0`, `lb`, `ub`, `plb`, `pub`, `non_box_cons`)

(faq-how-do-i-choose-the-starting-point-x0)=
### How do I choose the starting point `x0`?

First of all, keep in mind that you should restart PyBADS from different
starting points. Probably a minimum of ten, ideally dozens, depending on your
problem (see [the next question](#faq-how-do-i-run-pybads-from-several-starting-points)).

We recommend to choose starting points mostly inside the plausible box bounded
by `plb` and `pub`. Pass `x0=None` and PyBADS draws the starting point at random
inside the plausible box, uniformly (log-uniformly for a variable that PyBADS
[maps through a log](#faq-does-pybads-rescale-or-transform-my-variables)),
from the run's own random generator, so that the
[seed of the run](#faq-how-do-i-make-a-run-reproducible) decides it. Or draw
it yourself, for example

```python
rng = np.random.default_rng()
x0 = rng.uniform(plb, pub)
```

If you think that you would like to also draw points *outside* `plb` and
`pub`, then by definition it means that your choice of `plb` and `pub` is too
narrow (see also [below](#faq-how-do-i-choose-plb-and-pub)).

(faq-how-do-i-run-pybads-from-several-starting-points)=
### How do I run PyBADS from several starting points?

Create a new `BADS` object for each run: a `BADS` object runs a single
optimization. For example, with a random starting point and a different seed
for each run:

```python
n_runs = 10
results = []
for seed in range(n_runs):
    options = {"random_seed": seed, "display": "off"}
    bads = BADS(fun, None, lb, ub, plb, pub, options=options)
    results.append(bads.optimize())

best = min(results, key=lambda r: r["fval"])
x_best, fval_best = best["x"], best["fval"]
```

Then compare the runs. If several of them reach nearly the same `fval`, you
can be more confident about the solution; if they are scattered, see
[this question](#faq-i-have-been-running-pybads-with-a-deterministic-objective-function-from-the-same-starting-point-but-i-get-different-results-each-time-is-something-wrong)
for a deterministic objective and
[this one](#faq-i-have-been-running-pybads-with-a-stochastic-objective-function-from-different-starting-points-and-i-get-different-results-each-time-what-can-i-do)
for a noisy one. For a noisy objective, `fval` is itself an estimate, with
standard deviation `fsd`: runs whose values differ by less than a few `fsd`
cannot be told apart, but evaluating their solutions several more times can
tell them apart.

The runs are independent of each other, so you can also run them in
parallel processes, for instance with `concurrent.futures.ProcessPoolExecutor`,
provided that your objective function can be sent to another process.

(faq-how-do-i-choose-lb-and-ub)=
### How do I choose `lb` and `ub`?

`lb` and `ub` are the *hard bounds* of the optimization. In theory, you could
set them to the *mathematical* limits of your variables. However, using the
mathematical limits of a variable is a [bad](https://youtu.be/jyaLZHiJJnE?t=7s)
choice for optimization. Instead, we recommend to set them to no wider than
their *physical* or *experimental* limits.

For example, suppose that you have a parameter `sigma` that represents the
standard deviation (SD) of the movement endpoint of a subject in a task in
which people are asked to rapidly touch targets on a screen. *Mathematically*,
`sigma`, being a SD, could go from `0` to `np.inf`. However, in this case, it
is physically unrealistic that people would have no motor noise. Instead, we
set as `lb` our experimental lower bound, e.g., the resolution of our motion
tracker device, or maybe one screen pixel. Similarly, it is physically
impossible for people's pointing error to be larger than, say, the length of
their forearms. In fact, we could set as `ub` the size of the screen.

Importantly, do **not** set `lb` to `0` for variables that can only be
positive. Choose a small, *experimentally meaningful* number. Do **not** pick
extremely small numbers such as the machine epsilon,
`np.finfo(float).eps` (about `2e-16`), unless they are justified in the
context of your problem. A positive lower bound also lets PyBADS work on such
a variable in [log coordinates](#faq-does-pybads-rescale-or-transform-my-variables).

(faq-what-if-i-really-have-no-idea-how-to-choose-lb-and-ub)=
### What if I really have no idea how to choose `lb` and `ub`?

It is true that occasionally some model parameters might not have an a
priori *intuitive* range of values. One could gain a bit of intuition via
preliminary exploration of the function landscape (i.e., manually set some
values). Then, you could set bounds to some mid-to-large values, and expand
them if needed. We would still not recommend to set incredibly large bounds.

(faq-does-pybads-support-partially-unconstrained-optimization)=
### Does PyBADS support (partially) unconstrained optimization?

Yes and no. You can specify that a variable is (partially) unconstrained by
setting its hard bounds to `-np.inf` or `np.inf`; passing `None` for `lb` or
`ub` leaves every variable unbounded on that side. The plausible bounds of
such a variable must then be given, and finite.

However, we encourage users to always set finite, *empirically meaningful*
hard bounds (see [above](#faq-how-do-i-choose-lb-and-ub)). Infinities are never empirically
meaningful, unless perhaps
[if you are in a black hole](https://en.wikipedia.org/wiki/Cosmic_censorship_hypothesis).

(faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value)=
### Can I set `lb = ub` for some variable to fix it to a given value?

Yes: a variable whose four bounds, `lb`, `ub`, `plb` and `pub`, are equal
is fixed at that value, as in MATLAB BADS. Where `plb` and `pub` are not
given they are the hard bounds, so that `lb = ub` alone fixes the variable.
`x0` at a fixed variable is its value, or NaN, which stands for it. For
example, to fix the second of three variables at 2:

```python
lb = np.array([-5.0, 2.0, -5.0])
ub = np.array([5.0, 2.0, 5.0])
plb = np.array([-2.0, 2.0, -2.0])
pub = np.array([2.0, 2.0, 2.0])
x0 = np.array([0.0, 2.0, 0.0])

bads = BADS(fun, x0, lb, ub, plb, pub)  # optimizes x[0] and x[2]
optimize_result = bads.optimize()
```

PyBADS optimizes the other variables, as a run of the problem without the
fixed ones would, and the defaults of the options that depend on the number
of variables count only those: here `max_fun_evals` is `500 * 2`. When the
`BADS` object is created, PyBADS lists the fixed variables (from
`display="notify"` on). Your objective, `non_box_cons` and the
[output function](#faq-can-i-monitor-or-stop-a-run-while-it-is-running)
receive points of all the variables, with the fixed ones at their values,
and `optimize_result["x"]` and `optimize_result["x0"]`, the log of
evaluations (`bads.function_logger.X_orig`) and
`bads.iteration_history["x"]` hold them all. The indices of
`options["periodic_vars"]` count all the variables, and the points of
`precomputed_evaluations` hold them all. The run's
[internal state](#faq-where-can-i-find-the-internal-state-and-iteration-history)
(`bads.optim_state`, the internal coordinates `"u"` and the Gaussian
process) covers only the variables that are not fixed.

(faq-how-do-i-choose-plb-and-pub)=
### How do I choose `plb` and `pub`?

`plb` and `pub` are the *plausible* (or *reasonable*) bounds of the
optimization. Set them by thinking of a plausible range in which you would
expect to find almost all solutions; as a rule of thumb, you would bet that
the minimum lies inside the *plausible box* they define with probability
above 90%. The plausible box naturally represents a good region where to
randomly draw starting points for the optimization (see
[above](#faq-how-do-i-choose-the-starting-point-x0)). The plausible bounds
must be finite and satisfy `lb <= plb < pub <= ub` at every variable that
is not [fixed](#faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value).

If you *really* have no idea about a plausible range, you can set `plb` and
`pub` equal to `lb` and `ub`, or leave them out, in which case PyBADS uses
the hard bounds in their place, with the warning `bads:pbUnspecified`; but
this should not be the norm.

In the example above (see [this question](#faq-how-do-i-choose-lb-and-ub)),
the plausible bounds for `sigma` (pointing motor noise) could go from a few
pixels for `plb` to several cm for `pub`.

The plausible box also sets the scale of the optimization: PyBADS draws its
initial design of points inside it, and measures its steps in coordinates in
which the box spans `[-1, 1]` (see the [next question](#faq-does-pybads-rescale-or-transform-my-variables)).

(faq-does-pybads-rescale-or-transform-my-variables)=
### Does PyBADS rescale or transform my variables?

Yes, internally. Your objective always receives, and the result always
reports, points in the coordinates you gave, but PyBADS runs in coordinates
of its own:

- it maps the plausible box to `[-1, 1]` in each variable, so that its
  steps, and the `MeshScale` of its [display](#faq-what-are-the-quantities-displayed-by-pybads-during-optimization),
  are relative to the plausible range of each variable;
- before that, it takes the logarithm of every variable whose bounds are
  all positive (an infinite upper bound counts as positive) and whose
  plausible range spans a factor of 10 or more (`pub / plb >= 10`), unless
  the variable is
  [periodic](#faq-does-pybads-support-periodic-variables-such-as-angles).
  Its steps along such a variable are then proportional to the variable's
  value, which suits scale parameters such as standard deviations, rates or
  time constants, and a random starting point is drawn log-uniformly.
  PyBADS lists these variables when you create the `BADS` object.

To keep every variable on a linear scale, set
`options={"nonlinear_scaling": False}`. Otherwise, you rarely need to
rescale the variables yourself; a parameterization in which the parameters
trade off less against each other still makes the problem easier for any
optimizer.

(faq-how-do-i-prevent-pybads-from-evaluating-certain-inputs-or-regions-of-input-space)=
### How do I prevent PyBADS from evaluating certain inputs or regions of input space?

If these regions can be identified by coordinate-wise ranges, use `lb` and
`ub`. Otherwise, use the *barrier function* `non_box_cons` (non-box
constraints). It takes an array of shape `(N, D)`, one point per row, and
returns an array of `N` values, true (or positive) for each point that is not
allowed. PyBADS does not evaluate the objective at such points. For example,
to keep the optimization inside the unit ball:

```python
def non_box_cons(X):
    return np.sum(X**2, axis=1) > 1


bads = BADS(fun, x0, lb, ub, plb, pub, non_box_cons=non_box_cons)
```

The starting point `x0` must satisfy the constraints. See
[Example 2](https://acerbilab.github.io/pybads/_examples/pybads_example_2_nonbox_constraints.html)
for a full example.

PyBADS draws its initial points in the [plausible box](#faq-how-do-i-choose-plb-and-pub)
and leaves out those that violate the constraints, so choose a plausible box
of which a good part is allowed: in the example above, a box within `-1` and
`1` rather than a much wider one. A feasible region much thinner than the
steps of the optimization, such as a narrow band, can leave PyBADS unable to
find allowed points around `x0`, and the run then ends early: reparameterize
such a problem so that its feasible region is wide (the documentation of
[`BADS`](api/classes/bads.rst) gives an example). For the same reason,
PyBADS does not support equality constraints, such as `x[0] + x[1] = 1`:
write the problem in fewer variables, so that the constraint holds by
construction (here, optimize over `x[0]` alone and set `x[1] = 1 - x[0]`
inside the objective).

Do **not** have `fun(x)` return `np.inf` or `np.nan` for invalid inputs.
PyBADS would simply crash (see [this question](#faq-pybads-crashes-saying-that-the-returned-function-value-must-be-a-finite-real-valued-scalar-what-do-i-do)).

Absolutely do **NOT** have `fun(x)` return an arbitrarily large number to
enforce PyBADS to avoid certain regions. This strategy may seem innocent
enough, but in fact it completely cripples PyBADS by making its models of the
objective function nonsensical.

(faq-does-pybads-support-integer-constraints)=
### Does PyBADS support integer constraints?

No, PyBADS does not support integer constraints (that is, variables forced to
be integers).

As a simple fix, you could adopt the following hack if you have a single
integer variable `m`:

- make the parameter `m` continuous for the purpose of optimization;
- for a given parameter vector, evaluate separately the objective at
  `np.floor(m)` and `np.ceil(m)`;
- return the linearly interpolated value of the objective;
- at the end of the optimization, either return `np.round(m)`, or evaluate
  your function (several times, if noisy) at both `np.floor(m)` and
  `np.ceil(m)` and pick the best.

For example, with `m` the variable of index `i_int`:

```python
i_int = 0  # index of the integer variable


def fun_interp(x):
    m = x[i_int]
    x_lo, x_hi = x.copy(), x.copy()
    x_lo[i_int], x_hi[i_int] = np.floor(m), np.ceil(m)
    if x_lo[i_int] == x_hi[i_int]:
        return fun(x_lo)
    w = m - x_lo[i_int]
    return (1 - w) * fun(x_lo) + w * fun(x_hi)
```

Note that this will double the cost of each evaluation, so it is worth only
if alternative solutions (such as looping over all values of `m`) would be
computationally more expensive. Finally, this approach makes sense only if
`m` is truly an integer (ordered and with an underlying metric), as opposed
to merely a categorical variable.

(faq-does-pybads-support-periodic-variables-such-as-angles)=
### Does PyBADS support periodic variables, such as angles?

Yes. List the indices of the periodic variables, counted from 0, in the
option `periodic_vars`. The hard bounds `lb` and `ub` of a periodic variable
are its period, and need to be finite. PyBADS takes `lb` and `ub` as the
same point and wraps the variable around them, so that a run moves across
them as across any other value of the variable, and it models the objective
as periodic along the variable. PyBADS lists the periodic variables when you
create the `BADS` object.

The bounds of a periodic variable only mark where its period is cut, not a
range of extreme values, so its plausible bounds are usually its hard bounds
as well, unlike those of other variables
([How do I choose `plb` and `pub`?](#faq-how-do-i-choose-plb-and-pub)).
For example, with an angle in radians as the second variable:

```python
# x[0] is a position, x[1] an angle in radians
lb = np.array([-10.0, -np.pi])
ub = np.array([10.0, np.pi])
plb = np.array([-2.0, -np.pi])
pub = np.array([2.0, np.pi])

options = {"periodic_vars": [1]}
bads = BADS(fun, x0, lb, ub, plb, pub, options=options)
optimize_result = bads.optimize()
```

The result gives a periodic variable within its period, between `lb` and
`ub`. A minimum at `lb`, which is the same point as `ub`, can come out at
`lb`, just above it or just below `ub`, so two runs can report nearly the
same minimum at opposite ends of the period.

PyBADS raises a `ValueError` for a periodic variable with an infinite bound,
and for a `periodic_vars` that is not a list of distinct indices from 0 to
`D - 1`; give a boolean mask `m` as `np.flatnonzero(m)`.
[Example 6](https://acerbilab.github.io/pybads/_examples/pybads_example_6_periodic_variables.html)
runs PyBADS on a function with two periodic variables.

(faq-output-arguments)=
## Output arguments

(faq-what-does-optimize-return)=
### What does `optimize()` return?

`bads.optimize()` returns an [`OptimizeResult`](api/classes/optimize_result.rst),
a dictionary whose entries you read as `optimize_result["x"]` or as
attributes, `optimize_result.x`. Among them:

- `x`: the solution, a one-dimensional array;
- `fval`: the value of the objective at `x`, [estimated](#faq-how-is-fval-computed)
  for a noisy objective;
- `fsd`: the standard deviation of that estimate (`0` for a deterministic objective);
- `success`, `status` and `message`: why the run ended (see below);
- `func_count` and `iterations`: the number of function evaluations and of iterations;
- `total_time` and [`overhead`](#faq-how-is-overhead-defined).

The documentation of [`OptimizeResult`](api/classes/optimize_result.rst)
lists them all.

`success` is `True` when the run ended on one of its convergence criteria:
the mesh became smaller than `options["tol_mesh"]` (`status` 1), or the
objective improved by less than `options["tol_fun"]` over the last
`options["tol_stall_iters"]` iterations (`status` 2). It is `False`
(`status` 0) when the run used up its
budget of evaluations (`options["max_fun_evals"]`) or of iterations
(`options["max_iter"]`), or an [output function](#faq-can-i-monitor-or-stop-a-run-while-it-is-running)
stopped it. `message` says which. A run that used up its budget can still
have found a good solution; if not, give it [a larger budget](#faq-on-some-problems-pybads-seems-to-get-stuck-and-stop-too-early-is-there-a-way-to-tune-pybads-to-optimize-towards-a-higher-precision-result-or-to-have-it-optimize-for-longer).

(faq-how-is-fval-computed)=
### How is `fval` computed?

`fval` is the (estimated) value of the objective function at `x`, the
returned optimum.

For a deterministic (not-noisy) objective, this is simply `fun(x)`, the
lowest value that PyBADS observed. For a
[noisy objective](#faq-what-is-a-noisy-objective-function), PyBADS first
chooses `x` among the points at which each iteration ended, all estimated
again at the end with its Gaussian process model: the one whose estimate is
lowest after accounting for its uncertainty. It then evaluates `fun(x)`
several more times, `options["noise_final_samples"]` (default 10), and `fval`
is the average of those evaluations, and `fsd` its standard error. With
`options["specify_target_noise"]`, the average and its standard error weight
each evaluation by the precision that the objective reports. The evaluations
are in `optimize_result["yval_vec"]` (and the standard deviations that the
objective reported in `optimize_result["ysd_vec"]`). They count towards the
budget `options["max_fun_evals"]`, in which PyBADS sets them aside.

(faq-why-do-you-estimate-fval-by-averaging-additional-function-evaluations-cant-you-return-the-gaussian-process-prediction-at-x)=
### Why do you estimate `fval` by averaging additional function evaluations? Can't you return the Gaussian process prediction at `x`?

Glad that you asked. Yes, in theory we could use the Gaussian process (GP)
mean prediction at `x`. However, the GP prediction can occasionally fail,
sometimes subtly. While this mismatch is not a major problem *during*
optimization, it could potentially introduce hard-to-detect but substantial
biases in `fval`, which could have catastrophic effects for model selection.
For this reason, we chose a more conservative approach for estimating `fval`.

(faq-how-is-overhead-defined)=
### How is `overhead` defined?

`optimize_result["overhead"]` is the *fractional overhead*, defined as
(*total running time* / *total function time* - 1). The total running time,
`optimize_result["total_time"]`, is the time of `bads.optimize()`, in seconds;
the function time is the time spent evaluating the objective, except for the
second evaluation of `x0` that
[tests the objective for noise](#faq-should-i-tell-pybads-that-my-objective-is-noisy),
which counts as PyBADS's own time. PyBADS's own
work takes of the order of tens of milliseconds per function evaluation,
depending on the problem and the computer.

Typically, you would expect `overhead` to be (much) smaller than 1 for normal
runs of PyBADS. If the overhead is larger than, say, 0.75, your problem
affords fast evaluations and it is possible that it would benefit from other
algorithms than PyBADS (see [above](#faq-what-do-i-do-if-pybads-is-not-suited-for-my-problem)).

For PyBADS test problems and examples, you will find that the reported
overhead is astronomical, which is expected since for demonstration purposes
we are using simple analytical functions.

(faq-where-can-i-find-the-internal-state-and-iteration-history)=
### Where can I find the internal state and iteration history?

After `bads.optimize()`, the `BADS` object keeps:

- `bads.optim_state`, a dictionary with the state of the optimization;
- `bads.iteration_history`, which records each iteration: among others the
  point at which the iteration ended, in your coordinates (`"x"`) and in
  PyBADS's [internal coordinates](#faq-does-pybads-rescale-or-transform-my-variables)
  (`"u"`), its value (`"fval"` and `"fsd"`), the mesh size (`"mesh_size"`),
  the number of evaluations so far (`"func_count"`), and the Gaussian process
  model (`"gp"`), which works in the internal coordinates;
- `bads.function_logger`, the log of the evaluated points (see the
  [next question](#faq-is-it-possible-to-output-or-inspect-the-optimization-trajectory)).

These attributes are useful for debugging, but we are not providing explicit
support for them. Future versions of PyBADS might change their interface or
internal structure.

(faq-is-it-possible-to-output-or-inspect-the-optimization-trajectory)=
### Is it possible to output or inspect the optimization trajectory?

Yes. After `bads.optimize()`, the evaluated points and their observed values
are in the function log, whose rows `X_flag` marks as filled:

```python
log = bads.function_logger
X = log.X_orig[log.X_flag]  # evaluated points, one per row
y = log.Y_orig[log.X_flag].ravel()  # observed function values
```

Each row holds a distinct point, in the order in which PyBADS first
evaluated it. An evaluation at a point that is already in the log adds no
row: the second evaluation of `x0` that
[tests the objective for noise](#faq-should-i-tell-pybads-that-my-objective-is-noisy),
and the final evaluations of a noisy run, which are in
`optimize_result["yval_vec"]` (see [above](#faq-how-is-fval-computed)). So
the log can have fewer rows than `optimize_result["func_count"]`.

A run given evaluations made before it (the argument
`precomputed_evaluations` of `BADS`) holds them in the first rows of its
log, and does not count them in `func_count`. There a point can have more
than one row: with `uncertainty_handling=True` and without
`specify_target_noise`, each evaluation given has a row of its own, and,
unless `specify_target_noise` is set, the start adds a row where it
repeats a point given.

The points at which the iterations ended, and their values, are

```python
x_iter = np.vstack(bads.iteration_history["x"])
f_iter = bads.iteration_history["fval"]
```

To record the trajectory while the run goes on, use an
[output function](#faq-can-i-monitor-or-stop-a-run-while-it-is-running).

(faq-can-i-monitor-or-stop-a-run-while-it-is-running)=
### Can I monitor or stop a run while it is running?

Yes, with an output function, passed as `options["output_fcn"]`. PyBADS
calls it as `output_fcn(x, optim_state, state)` once the initial points
have been evaluated (`state` is `"init"`), at the end of each iteration
(`"iter"`) and when the run ends (`"done"`). `x` is the current best point, in your coordinates, and
`optim_state` a copy of the state of the optimization, with for instance
`optim_state["fval"]`, `optim_state["fsd"]` and `optim_state["mesh_size"]`.
When the function returns `True`, the run stops, and its result says so in
`message`, with `success` set to `False`.

For example, to record the point at the end of each iteration, and to stop
the run after an hour:

```python
import time

start = time.perf_counter()
trace = []


def output_fcn(x, optim_state, state):
    if state == "iter":
        trace.append((x.copy(), optim_state["fval"]))
    return time.perf_counter() - start > 3600


bads = BADS(fun, x0, lb, ub, plb, pub, options={"output_fcn": output_fcn})
optimize_result = bads.optimize()
```

(faq-noisy-objective-function)=
## Noisy objective function

(faq-what-is-a-noisy-objective-function)=
### What is a *noisy* objective function?

A *noisy* (or *stochastic*) objective function is an objective that will
return different results if evaluated twice at the same point `x`. A
non-noisy objective function is *deterministic*.

For model fitting, objective functions can be noisy if the log likelihood is
evaluated through simulation (e.g., via Monte Carlo methods).

(faq-why-are-noisy-objective-functions-treated-differently)=
### Why are noisy objective functions treated differently?

For a deterministic objective, we assume that the goal is to minimize *f(x)*.
For a noisy objective, we assume that the goal is to minimize the *expected
value* of *f(x)*, also written as E[*f(x)*]. For this reason, PyBADS will not
simply blindly trust whatever *f(x)* returns, but will do some internal
computation to estimate E[*f(x)*] (effectively, smoothing the observed
function values via a Gaussian process).

Incidentally, this means that ideally the function that you provide (and
that computes the negative log likelihood) should be an *unbiased* estimator
of the negative log likelihood, but this is a story for another time.

(faq-can-i-make-a-noisy-objective-function-deterministic-by-fixing-the-noise-process)=
### Can I make a noisy objective function deterministic by fixing the noise process?

Well, technically yes, you *could* make a noisy objective function
deterministic by seeding its random number generator again (e.g., with
`np.random.default_rng(0)`) every time you call it. However, this fix does
not really solve the problem, because you are not eliminating the noise in
the function observations. In fact, if you do it naively, you might be adding
unwanted bias to your fits.

Thus, it is *not* recommended to 'fix' the noise this way (by fixing the
random seed at each function call). Instead, let your function be
stochastic, and let PyBADS deal with it.

Note that this is different from setting the random seed *once* at the
beginning of an optimization run, for the sake of reproducibility, which is
recommended as good practice. The seed of PyBADS,
[`options["random_seed"]`](#faq-how-do-i-make-a-run-reproducible), governs
only the random draws of PyBADS itself, so give your objective a random
generator of its own, created once:

```python
rng = np.random.default_rng(12345)  # created once, for the whole run


def fun(x):
    return simulated_nll(x, rng)  # draws new noise at every call
```

where `simulated_nll` stands for your simulation-based estimate of the
negative log likelihood.

(faq-should-i-tell-pybads-that-my-objective-is-noisy)=
### Should I tell PyBADS that my objective is noisy?

Please do so. Set `options={"uncertainty_handling": True}` to tell PyBADS
that the optimization is noisy.

If you forget about it, PyBADS will determine at initialization whether the
provided objective is noisy, by evaluating it twice at `x0` (an evaluation
that counts towards the budget). Note that this test can fail if a noisy
objective happens to return the same value twice, which is rare unless its
values take only a few distinct levels.

Conversely, set `options={"uncertainty_handling": False}` for a
deterministic objective: PyBADS then skips the test, and saves an
evaluation. With `options={"specify_target_noise": True}` (see
[below](#faq-should-i-provide-an-estimate-of-the-noise-associated-with-each-evaluation)),
PyBADS treats the objective as noisy without further ado.

(faq-can-pybads-handle-any-arbitrary-amount-of-noise-in-the-objective)=
### Can PyBADS handle any arbitrary amount of noise in the objective?

No. PyBADS works best if the standard deviation of the objective function,
when evaluated in the vicinity of the solution, is small with respect to
changes in the objective function itself (that is, there is a good
signal-to-noise ratio). In many cases, a standard deviation of order `1` or
less should work (this is the default assumption). If you approximately know
the magnitude of the noise in the vicinity of the solution, you can help
PyBADS by specifying it in advance (set `options["noise_size"] = sigma_est`,
where `sigma_est` is your estimate of the standard deviation).

If the noise around the solution is too large, PyBADS will perform poorly.
In that case, we recommend to increase the precision of your computation of
the objective (e.g., by drawing more Monte Carlo samples) such that
`sigma_est` is of order 1 or even lower, as needed by your problem (see also
[this related question](#faq-i-have-been-running-pybads-with-a-stochastic-objective-function-from-different-starting-points-and-i-get-different-results-each-time-what-can-i-do)).
Note that the noise farther away from the solution can be larger, and this is
usually okay.

(faq-does-pybads-assume-that-the-noise-is-the-same-for-all-inputs)=
### Does PyBADS assume that the noise is the same for all inputs?

Yes and no. The Gaussian process (GP) model built by PyBADS is
*homoskedastic*, that is, it assumes constant noise across the input space.
However, the GP model is built using only a *local* set of points, so PyBADS
will adapt to local characteristics of the objective function, including
amounts of noise that depend on the location.

You can help PyBADS optimize a *heteroskedastic* objective (i.e., with
input-dependent noise) by providing an estimate of the noise at each
location, as specified in the questions below.

(faq-should-i-provide-an-estimate-of-the-noise-associated-with-each-evaluation)=
### Should I provide an estimate of the noise associated with each evaluation?

Yes, you should if you can! This may considerably help PyBADS, especially if
the objective is particularly noisy or strongly heteroskedastic. Remember to:

- set `options["specify_target_noise"] = True`;
- pass to PyBADS a function `fun` that returns a tuple `(f, sd)`: the
  estimate of the objective at `x`, and an estimate of its standard
  deviation (SD), a finite positive number.

```python
def fun(x):
    f, sd = estimate_nll(x)  # your estimate and its SD
    return f, sd


bads = BADS(fun, x0, lb, ub, plb, pub, options={"specify_target_noise": True})
```

Without `specify_target_noise`, PyBADS refuses such a pair: the objective
must return a single number. See
[Example 4](https://acerbilab.github.io/pybads/_examples/pybads_example_4_user_provided_noise.html)
for further information.

(faq-how-do-i-estimate-the-standard-deviation-of-a-noisy-objective)=
### How do I estimate the standard deviation of a noisy objective?

If you use [*inverse binomial sampling* (IBS)](https://github.com/acerbilab/ibs),
the algorithm returns the variability of the estimate as second output. Just
ensure that the variability is returned as *standard deviation* (SD) and not
as the variance (depending on the implementation, you may have to take the
square root of the reported variance). Otherwise, you can also estimate the
SD via bootstrap or similar approaches.

(faq-display)=
## Display

(faq-what-are-the-quantities-displayed-by-pybads-during-optimization)=
### What are the quantities displayed by PyBADS during optimization?

With `options["display"]` set to `"iter"` (the default), PyBADS displays the
traces of several optimization quantities:

- the `Iteration` number;
- the number of objective function evaluations `f-count`;
- the value of `f(x)` at the *incumbent* (current point);
- the normalized `MeshScale` size (that is, the POLL size parameter
  normalized to the [*plausible box*](#faq-how-do-i-choose-plb-and-pub));
- the current optimization stage and method, under `Method`: `Initial mesh`
  for the initial design; a `Successful search`, which improved on the
  incumbent sufficiently, or an `Incremental search`, which improved on it
  by less, with the search method in parentheses (`ES-wcm` or `ES-ell`); a
  `Successful poll`; or `Refine grid`, a poll that found no sufficient
  improvement (it can still move to a slightly better point), after which
  the mesh shrinks;
- additional actions, under `Actions`, such as the `Uncertainty test` of the
  objective's noise at `x0`, the evaluation of the `Initial points`, or
  re-training the Gaussian process (`Train`).

If the objective function is [noisy](#faq-noisy-objective-function), instead
of `f(x)` PyBADS will report the expected value `E[f(x)]` and its standard
deviation `SD[f(x)]` at the incumbent, both estimated via the current
Gaussian process model. The first lines of the display come before these
estimates: there `E[f(x)]` is the value observed at the incumbent, and
`SD[f(x)]` is `nan`, or a nominal value (`options["noise_size"]`, or the
standard deviation that the objective returned).

(faq-for-a-noisy-function-i-noticed-that-the-series-of-displayed-efx-values-is-not-monotonically-decreasing-should-i-worry)=
### For a noisy function, I noticed that the series of displayed `E[f(x)]` values is *not* monotonically decreasing. Should I worry?

Well spotted, but nothing to worry about. PyBADS keeps updating the estimate
of `E[f(x)]` at the incumbent, which means that occasionally this value will
*increase* across iterations, and sometimes it will oscillate for a few
iterations. Also, if the noise is large, the Gaussian process approximation
might occasionally fail, leading to outlier estimates for `E[f(x)]` (which
should then recover in the subsequent iterations). All of this is part of the
normal functioning of PyBADS.

(faq-for-a-noisy-function-i-noticed-that-the-series-of-displayed-sdfx-sometimes-shows-sudden-jumps-eg-from-1-to-4-is-that-normal)=
### For a noisy function, I noticed that the series of displayed `SD[f(x)]` sometimes shows sudden jumps (e.g., from ~1 to ~4). Is that normal?

First, recall that `SD[f(x)]` is the estimated posterior standard deviation
of the objective function at the incumbent (current best point). This
estimate is obtained via the Gaussian process model built by PyBADS every few
iterations. When the incumbent changes, or when PyBADS re-trains the Gaussian
process model, the uncertainty about the value at the incumbent will also
change, sometimes substantially (e.g., if the estimated observation noise
parameter has changed, or if the incumbent has moved to a new region). In
most cases, such jumps are part of the normal behavior of the algorithm.

(faq-sometimes-as-actions-during-optimization-i-read-train-failed-what-does-it-mean)=
### Sometimes as `Actions` during optimization I read `Train (failed)`. What does it mean?

It means that PyBADS was unable to refit the Gaussian process model to the
current local training set, usually due to numerical issues. Occasional
failures are not reason of concern, in particular at the beginning or towards
the end of the optimization. However, if a large number of training attempts
are systematically failing, it might mean that PyBADS is having trouble.
Sometimes this can be fixed by changing the problem parameterization, or
perhaps there are other issues with the model.

(faq-how-do-i-silence-pybads-or-send-its-output-elsewhere)=
### How do I silence PyBADS, or send its output elsewhere?

`options["display"]` sets how much PyBADS prints: `"iter"` (the default)
prints a line per step, as described [above](#faq-what-are-the-quantities-displayed-by-pybads-during-optimization);
`"final"` prints the opening and final messages; `"notify"` the opening
messages alone; `"off"` nothing but warnings; `"full"` everything, debug
messages included. With `"iter"` or `"full"`, a run may also print a short
tip before its first iteration line, with a link to the documentation: the
first such run of a Python session, then every third, each tip at most
once per session. `options["show_tips"] = False` turns the tips off.

PyBADS prints through Python's `logging` module, with a logger named
`"BADS"`, and its warnings, such as `bads:pbUnspecified`, are log messages
too, not Python warnings (the libraries it calls, such as NumPy and gpyreg,
can still issue Python warnings of their own). Creating a `BADS` object
calls `logging.basicConfig`, which sends the messages to the standard output
unless your program has configured `logging` before; a `logging.basicConfig`
call of your own after that takes effect only with `force=True`. To write
PyBADS's messages to a file instead:

```python
import logging

logger = logging.getLogger("BADS")
logger.addHandler(logging.FileHandler("bads_run.log"))
logger.propagate = False  # not to the console as well
```

(faq-troubleshooting)=
## Troubleshooting

(faq-is-there-a-way-to-check-that-pybads-is-running-correctly-is-it-enough-that-it-does-not-give-warningserrors)=
### Is there a way to check that PyBADS is running correctly — is it enough that it does not give warnings/errors?

You can follow the run through its [display](#faq-display), and inspect its
state while it runs with an [output function](#faq-can-i-monitor-or-stop-a-run-while-it-is-running).
And while the fact that no warnings/errors are shown is *encouraging*, it is
not a sufficient condition to guarantee that everything ran correctly. For
validation of the results, we recommend usual techniques such as comparing
multiple independent runs of the algorithm, and various form of model
checking.

(faq-pybads-crashes-saying-that-the-returned-function-value-must-be-a-finite-real-valued-scalar-what-do-i-do)=
### PyBADS crashes saying that `The returned function value must be a finite real-valued scalar`. What do I do?

This `ValueError` means that your objective function has returned `np.inf`,
`np.nan`, or something that is not a single real number, such as a complex
number or an array of several elements. You should check your code and
understand why it returned such a value. A pair `(f, sd)` gives this error
too, unless you set `options["specify_target_noise"] = True` (see
[this question](#faq-should-i-provide-an-estimate-of-the-noise-associated-with-each-evaluation)).

`inf`s and `nan`s often arise because there are outcomes in your dataset
(e.g., responses in a trial) to which the tested model assigns a probability
of `0` (usually due to numerical truncation). `np.log(0)` yields `-np.inf`,
which is then propagated. In these cases, we recommend to make the model more
robust, by computing the likelihood directly in log space where you can (for
instance with log-density functions such as `scipy.stats.norm.logpdf`, and
`scipy.special.logsumexp`), or by forcing all outcomes (e.g., the likelihood
associated with each trial) to have a minimum non-null probability, such as
`np.sqrt(np.finfo(float).eps)` or some other small value. This should not be
necessary if the model already includes a non-zero *lapse rate*.

`nan`s also arise when you take `np.log` or `np.sqrt` of a negative number
(of a quantity that should not be negative), which in NumPy gives `nan` with
a `RuntimeWarning`. You might be setting wrong bounds for your variables, or
maybe there are indexing issues.

Note that some optimizers are robust to `inf`s and `nan`s and just keep
going, avoiding the problematic region. However, we believe this is dangerous
as it might hide deeper issues with the model implementation.

(faq-during-optimization-i-received-a-warning-that-the-mesh-attempted-to-expand-above-maximum-size-too-many-times-what-does-it-mean)=
### During optimization I received a warning that `The mesh attempted to expand above maximum size too many times`. What does it mean?

It means that *probably* your `plb` or `pub` bounds are too narrow; try
widening them. If these are already as wide as `lb` and `ub`, it might be
that your hard bounds are too narrow.

If you do not think that this is the case, you can disable this warning by
setting `options["mesh_overflow_warning"] = np.inf`.

(faq-i-am-passing-non_box_cons-to-pybads-but-i-get-an-error-that-non_box_cons-should-be-a-function-that-takes-an-n-x-d-array-x-what-am-i-doing-wrong)=
### I am passing `non_box_cons` to PyBADS, but I get an error that `non_box_cons should be a function that takes an N x D array X`. What am I doing wrong?

Most likely, your `non_box_cons` accepts only a *single point* and returns a
single value, whereas you should be sure that it takes an array of shape
`(N, D)`, one point per row, and returns an array of `N` constraint
violations.

For example, suppose that your input variables need to be ordered, such that
`x[0] <= x[1]` and `x[1] <= x[2]`. Then you should set

```python
def non_box_cons(X):
    return (X[:, 0] > X[:, 1]) | (X[:, 1] > X[:, 2])
```

whereas `lambda x: x[0] > x[1] or x[1] > x[2]` would yield an error.

(faq-i-have-been-running-pybads-with-a-deterministic-objective-function-from-the-same-starting-point-but-i-get-different-results-each-time-is-something-wrong)=
### I have been running PyBADS with a *deterministic* objective function from the *same* starting point, but I get different results each time. Is something wrong?

Nothing is wrong per se. PyBADS is a stochastic optimizer, so results may
differ between different runs, even with the same initial condition `x0`. If
the returned `fval` varies *substantially* across runs from the same starting
point, it might be a sign that your function landscape is particularly
difficult. If `fval` is similar across runs, but the returned optimum `x`
varies substantially, it is a sign that your function has a plateau or
ridge, with trade-offs between parameters.

If you want to have reproducible results (and this advice applies beyond
PyBADS), we recommend to fix the random seed of each run to some known
quantity (e.g., setting it to the run number), as explained in the
[next question](#faq-how-do-i-make-a-run-reproducible).

(faq-how-do-i-make-a-run-reproducible)=
### How do I make a run reproducible?

Pass an integer seed when you create the `BADS` object:

```python
bads = BADS(fun, x0, lb, ub, plb, pub, options={"random_seed": 42})
```

Every random draw of the run comes from one NumPy random generator created
from this seed, `bads.rng`: the random starting point when `x0` is `None`,
the initial design, the searches, the poll and the fits of the Gaussian
process. With the same seed, objective, inputs and options, a run gives the
same result every time on the same computer, with the same versions of
Python, NumPy, SciPy and gpyreg and the same number of threads for linear
algebra; another computer, other versions or another number of threads can
give a different result. On Apple Silicon Macs, two runs with the same seed
make the same random draws but can end at slightly different points: with
Apple's Accelerate as the linear algebra library of NumPy and SciPy (as in
their wheels on PyPI), the last bits of a result depend on where its arrays
lie in memory. The seed is read when the `BADS` object is created, and
`optimize_result["random_seed"]` records it.

If you leave `random_seed` unset, PyBADS derives the generator of the run
from NumPy's global random state, so `np.random.seed(42)` before creating the
`BADS` object also fixes the run. PyBADS otherwise neither draws from nor
seeds that global state.

Whether a run prints a
[tip](#faq-how-do-i-silence-pybads-or-send-its-output-elsewhere), and which,
depends on the runs before it in the Python session, not on the seed; a tip
does not affect the run.

The seed does not govern your objective: if your objective is noisy, give it
a random generator of its own, as explained
[above](#faq-can-i-make-a-noisy-objective-function-deterministic-by-fixing-the-noise-process).
A seed of PyBADS does not reproduce a run of MATLAB BADS either, since the
two draw their random numbers differently.

(faq-i-have-been-running-pybads-with-a-stochastic-objective-function-from-different-starting-points-and-i-get-different-results-each-time-what-can-i-do)=
### I have been running PyBADS with a *stochastic* objective function from different starting points and I get different results each time. What can I do?

First of all, *slightly* different results are expected if your objective
function is noisy (be sure to have read and understood all the points under
the [noisy objective function](#faq-noisy-objective-function) section of the
FAQ). So, if you are asking this question, it is because you find *wildly*
different results.

Generally, substantially different results suggest that PyBADS is getting
stuck due to excess noise in the objective function with respect to actual
improvements of the function in the neighborhood of the current point. For
example, even if the expected value of the objective function would have a
non-zero gradient, it might be too hard for PyBADS to find a direction of
improvement due to low signal-to-noise ratio. In particular, because of a
slight conservative bias of the algorithm under uncertainty (needed to avoid
chasing random fluctuations), both the poll and search steps repeatedly fail
to find a significant improvement, and thus the algorithm stops moving. For
this reason, it is possible for PyBADS to get stuck at very different points
in noisy, nearly-flat regions of the input space, with more scattered results
for flatter and wider plateaus.

The general solution of this problem, as also mentioned
[in this question](#faq-can-pybads-handle-any-arbitrary-amount-of-noise-in-the-objective),
is to decrease the amount of noise in the objective function (e.g., if you
estimate the objective function via Monte Carlo sampling, try increasing the
number of samples). While we generally found that a SD of the noise of 1 or
less in the vicinity of the solution works for most problems, a particularly
difficult (e.g., flat) objective function might need even lower amounts of
noise for robust convergence, so
[YMMV](https://www.urbandictionary.com/define.php?term=ymmv).

(faq-on-some-problems-pybads-seems-to-get-stuck-and-stop-too-early-is-there-a-way-to-tune-pybads-to-optimize-towards-a-higher-precision-result-or-to-have-it-optimize-for-longer)=
### On some problems, PyBADS seems to get stuck and stop too early. Is there a way to tune PyBADS to optimize towards a higher precision result or to have it optimize for longer?

First, check why the run ended, in `optimize_result["message"]`. If it used
up its budget, raise `options["max_fun_evals"]` (default `500 * D`
evaluations) or `options["max_iter"]` (default `200 * D` iterations).

Otherwise, there are some options one can modify in PyBADS to make the
algorithm poll or search for longer. This might help *in some cases*
(especially for noisy objectives). If you want PyBADS to search for longer
at each iteration, you can modify two key options in the `options`
dictionary that you pass to the algorithm:

- Set `options["complete_poll"] = True` (default is `False`). This will force
  PyBADS to finish the "poll" step (more info in the paper) instead of
  skipping it when it thinks that it is not worth continuing. This is a
  basic option of PyBADS.
- Change `options["search_n_try"]`. Be careful that this is an advanced
  option of PyBADS, and we do not recommend to change it unless you have to.
  The default value is `max(D, floor(3 + D/2))`, where `D` is the number
  of variables that PyBADS optimizes, all but the
  [fixed](#faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-given-value)
  ones. This quantity represents the
  number of searches (via local Bayesian optimization) that PyBADS attempts
  in each round of searches before a poll. You can try and increase it to
  force PyBADS to search for longer in each iteration.

(faq-on-some-problems-pybads-seems-to-find-a-reasonably-good-solution-but-then-it-takes-a-long-time-to-converge-spending-many-iterations-at-very-small-values-of-meshscale-is-there-a-way-to-tune-pybads-to-stop-earlier-once-it-finds-a-decent-solution)=
### On some problems, PyBADS seems to find a reasonably good solution, but then it takes a long time to converge, spending many iterations at very small values of `MeshScale`. Is there a way to tune PyBADS to stop earlier once it finds a decent solution?

This issue is relatively common with [noisy objective functions](#faq-noisy-objective-function).
If it seems that PyBADS is spending too much time to converge at small mesh
scales, e.g., dozens and dozens of iterations spent at `MeshScale` below
`0.0001` or so, with little changes to the value of `f(x)` (or `E[f(x)]` for
noisy objectives) consider modifying `options["tol_mesh"]`.

This option determines a bunch of PyBADS behaviors, including the termination
condition when the normalized `MeshScale` goes below a threshold. The default
value is `options["tol_mesh"] = 1e-6`, which may be exceedingly conservative
for stochastic problems. You could try larger values, such as
`options["tol_mesh"] = 1e-5` or even `options["tol_mesh"] = 1e-4`. However,
use higher values at your own risk, in that you might be terminating a run
which is still improving on the solution.

(faq-miscellanea)=
## Miscellanea

(faq-this-is-interesting-but-shouldnt-we-ideally-compute-full-posterior-distributions)=
### This is interesting, but shouldn't we ideally compute full posterior distributions?

Yes, we should! However, given the typical class of model-fitting problems
PyBADS is designed for (see [here](#faq-which-kind-of-problems-is-pybads-suited-for)),
obtaining the full posterior, or even an approximation thereof, can be a
challenging task. We developed a method and related toolbox,
[Variational Bayesian Monte Carlo (PyVBMC)](https://acerbilab.github.io/pyvbmc/),
which addresses exactly this problem. Check it out!

(faq-can-pybads-return-an-approximate-posterior-eg-by-computing-the-hessian-at-the-optimum)=
### Can PyBADS return an approximate posterior, e.g. by computing the Hessian at the optimum?

First, just to clarify, the Hessian is a matrix of second derivatives, which
can be used to build a crude approximation of the posterior via
[Laplace's method](http://www.inference.org.uk/mackay/itprnn/ps/341.342.pdf).
The answer to the question is nope, PyBADS cannot return the Hessian or an
approximate posterior. The reason is that even if PyBADS tries to build a
local Gaussian process approximation of the objective function, this might
fail and we cannot trust this approximation at all to represent a valid
posterior.

Instead, you should look into
[Variational Bayesian Monte Carlo (PyVBMC)](https://acerbilab.github.io/pyvbmc/),
a method that we developed specifically to compute approximate posterior
distributions, and that can be used in synergy with PyBADS (see
[below](#faq-i-have-run-pybads-on-my-problem-how-do-i-run-pyvbmc)).

(faq-i-have-run-pybads-on-my-problem-how-do-i-run-pyvbmc)=
### I have run PyBADS on my problem. How do I run PyVBMC?

[PyVBMC](https://acerbilab.github.io/pyvbmc/) computes an approximate
posterior distribution over the parameters, and an estimate of the model
evidence (see [above](#faq-this-is-interesting-but-shouldnt-we-ideally-compute-full-posterior-distributions)).
Its interface is very similar to the one of PyBADS, and you may only need
minor changes to run it on your problem. Note that:

- Beware of the sign! The target of PyVBMC is a log density, that is the log
  likelihood, or the log likelihood plus the log prior, whereas PyBADS
  *minimizes* the negative log likelihood.
- PyVBMC needs a prior over the parameters, which you pass with `prior=` or
  add to the log likelihood.
- The plausible bounds of PyVBMC should lie strictly inside its hard bounds,
  whereas PyBADS takes plausible bounds equal to the hard ones.
- PyVBMC does not support variables bounded on one side only, nor fixed
  variables: give it a function of the other variables that inserts the
  fixed values (`np.insert(x, i, value)`), with the bounds of the other
  variables.
- The solution of PyBADS is a good starting point `x0` for PyVBMC.
- PyVBMC supports noisy targets too, and works best when the target returns
  an estimate of its noise.

For example, with a uniform prior over the hard bounds:

```python
from pyvbmc import VBMC
from scipy.stats import uniform


def log_likelihood(x):
    return -fun(x)  # fun is the negative log likelihood minimized by PyBADS


prior = [
    uniform(loc=low, scale=high - low)
    for low, high in zip(np.ravel(lb), np.ravel(ub))
]
vbmc = VBMC(
    log_likelihood, optimize_result["x"], lb, ub, plb, pub, prior=prior
)
vp, results = vbmc.optimize()
```

See the [PyVBMC documentation](https://acerbilab.github.io/pyvbmc/) for
installation, priors and examples.

(faq-i-used-bads-in-matlab-what-is-different-in-pybads)=
### I used BADS in MATLAB. What is different in PyBADS?

PyBADS implements the same algorithm, with a Python interface:

- You create a `BADS` object and run it,
  `optimize_result = BADS(fun, x0, lb, ub, plb, pub, non_box_cons, options).optimize()`,
  where MATLAB returns `[X,FVAL,EXITFLAG,OUTPUT] = bads(...)`. The result is
  one dictionary, with `x`, `fval`, `status` (MATLAB's `EXITFLAG`),
  `message`, `func_count` and more (see [above](#faq-what-does-optimize-return)).
- The options are a dictionary. Their names are mostly the MATLAB names in
  lower case, with underscores (`MaxFunEvals` becomes `max_fun_evals`,
  `UncertaintyHandling` becomes `uncertainty_handling`), and their values are
  Python values: numbers where MATLAB takes a string such as `'500*nvars'`,
  `True` or `False` where MATLAB takes `1`, `0`, `'on'` or `'off'`, and
  indices counted from 0 where MATLAB counts from 1 (`periodic_vars` is
  `[2, 3]` where MATLAB's `PeriodicVars` is `[3 4]`).
  PyBADS refuses an option name it does not know with a `ValueError`, and
  checks the values of many options when `BADS` is created: a value of the
  wrong kind, such as the string `'200*D'` for `max_iter`, raises a
  `ValueError` that names the option. A wrong value of another option can
  instead fail with another error, for some options only once the run has
  started. The [options page](api/options/bads_options.rst) lists them all.
- The objective receives a one-dimensional array of shape `(D,)`, and
  `non_box_cons` an array of shape `(N, D)`. Additional inputs of the
  objective are [bound to it](#faq-my-objective-function-requires-additional-datainputs-how-do-i-pass-them-to-pybads)
  rather than passed to `BADS`.
- A few options of MATLAB BADS, such as `plot` and `restarts`, are accepted
  but have no effect.
- Function evaluations made before the run, which MATLAB BADS takes in its
  option `FunValues`, are passed to `BADS` as the argument
  `precomputed_evaluations=(X, y)`, or `(X, y, y_sd)` with
  `specify_target_noise`. They enter the run's log and its Gaussian process
  but not its count of evaluations, and the run starts from `x0` and its
  initial design, as in MATLAB BADS.
- Runs of PyBADS and of MATLAB BADS do not match step by step, even with the
  same seed. The
  [catalogue of differences](https://github.com/acerbilab/pybads/blob/main/pybads/bads/README.md)
  lists where PyBADS deliberately departs from MATLAB BADS, and why.

(faq-are-you-planning-to-port-bads-to-other-languages)=
### Are you planning to port BADS to other languages?

BADS is currently available as a [MATLAB toolbox](https://github.com/acerbilab/bads)
and as a [Python package](https://github.com/acerbilab/pybads), PyBADS. No
other ports are currently planned, but please get in touch if interested.
The Python package can also be called from other languages, for example from
Julia with [PythonCall.jl](https://cjdoris.github.io/PythonCall.jl/stable/)
or from R with [reticulate](https://rstudio.github.io/reticulate/).
