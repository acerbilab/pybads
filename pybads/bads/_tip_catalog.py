"""Data-only catalogue of the runtime tips that a run may show."""

from dataclasses import dataclass
from typing import Literal

TipFrequency = Literal["normal", "low_frequency"]

_FAQ = "https://acerbilab.github.io/pybads/faq.html"
_HUB = "https://acerbilab.org/model-fitting/"


@dataclass(frozen=True)
class Tip:
    """One runtime tip: a stable id, its text, how often it may show, and the
    links that it gives the reader."""

    id: str
    text: str
    frequency: TipFrequency
    urls: tuple = ()


# To edit a tip's wording or links, keep its id; give a new topic a new id.
# Tips are added, removed or moved between the frequencies here alone: the
# scheduler (_runtime_tips.py) reads no size or id of this catalogue. The
# text is plain ASCII, which any console encodes. The URLs are those of the
# published documentation and, for a tip that names another of the lab's
# methods, the lab's page of its tools for fitting models to data (_HUB).
# Each tip restates in brief the advice of the answer or the page that it
# links, so the FAQ's labels, that advice and that page's description of
# the method are coupled with this catalogue (AGENTS.md).
TIPS = (
    Tip(
        id="multiple_starts",
        text=(
            "Run PyBADS from at least 10 different starting points, ideally "
            "dozens, depending on your problem; with x0=None, each run draws "
            "its start at random in the plausible box. Keep the run with the "
            "lowest fval: a single run can stop in a local minimum, and if "
            "several runs reach nearly the same fval, you can be more "
            "confident in the solution."
        ),
        frequency="normal",
        urls=(f"{_FAQ}#faq-how-do-i-run-pybads-from-several-starting-points",),
    ),
    Tip(
        id="plausible_bounds",
        text=(
            "Set plb and pub around where you expect the solution, a box you "
            "would bet contains it with over 90% probability. PyBADS draws "
            "its first points inside that box and scales its steps by it; lb "
            "and ub only bound the search."
        ),
        frequency="normal",
        urls=(f"{_FAQ}#faq-how-do-i-choose-plb-and-pub",),
    ),
    Tip(
        id="noisy_target",
        text=(
            "If your objective is noisy, for instance a negative "
            "log-likelihood estimated by simulation, set "
            "options['uncertainty_handling'] = True rather than relying on "
            "PyBADS's noise test at x0. In many cases a noise SD of about 1 "
            "or less near the solution works; if it is larger, reduce it, for "
            "instance with more simulations per evaluation. If you can "
            "estimate the SD of each evaluation, set "
            "options['specify_target_noise'] = True and return (f, sd)."
        ),
        frequency="normal",
        urls=(f"{_FAQ}#faq-noisy-objective-function",),
    ),
    Tip(
        id="fixed_variables",
        text=(
            "To hold a parameter at a known value, set its four bounds (lb, "
            "ub, plb and pub) to that value, and its entry of x0 to the same "
            "value or NaN. PyBADS optimizes only the other parameters and "
            "still passes all of them to your objective, so your code needs "
            "no change."
        ),
        frequency="normal",
        urls=(
            f"{_FAQ}#faq-can-i-set-lb-ub-for-some-variable-to-fix-it-to-a-"
            "given-value",
        ),
    ),
    Tip(
        id="periodic_vars",
        text=(
            "For angles and other periodic parameters, list their indices "
            "(from 0) in options['periodic_vars'], with the period as their "
            "hard bounds (e.g. -np.pi and np.pi for an angle in radians), "
            "usually as their plausible bounds too. PyBADS then treats the "
            "two bounds as the same point and can move across them."
        ),
        frequency="normal",
        urls=(
            f"{_FAQ}#faq-does-pybads-support-periodic-variables-such-as-"
            "angles",
        ),
    ),
    Tip(
        id="stop_reason",
        text=(
            "The message at the end of the run, also in "
            "optimize_result['message'], says why it ended. If it used up its "
            "budget (max_fun_evals, 500 * D by default, D the number of "
            "parameters, or max_iter) before settling, raise it; if it converged but the solution seems "
            "imprecise, the FAQ lists options that make it search longer."
        ),
        frequency="normal",
        urls=(
            f"{_FAQ}#faq-on-some-problems-pybads-seems-to-get-stuck-and-stop-"
            "too-early-is-there-a-way-to-tune-pybads-to-optimize-towards-a-"
            "higher-precision-result-or-to-have-it-optimize-for-longer",
        ),
    ),
    Tip(
        id="pyvbmc",
        text=(
            "If your objective is a negative log-likelihood, you can also "
            "estimate the uncertainty over the parameters and the model "
            "evidence with PyVBMC, another of the lab's model-fitting tools "
            "(best with up to about 10 parameters). You can run it on the "
            "same model and data, with a prior over the parameters and "
            "PyBADS's solution as its starting point x0."
        ),
        frequency="low_frequency",
        urls=(
            _HUB,
            f"{_FAQ}#faq-i-have-run-pybads-on-my-problem-how-do-i-run-pyvbmc",
        ),
    ),
    Tip(
        id="agent_skill",
        text=(
            "If you work with a coding agent, give it the PyBADS skill from "
            "GitHub, or copy the skills/pybads folder of PyBADS's repository "
            "into the agent's skill directory: it points the agent to the "
            "parts of PyBADS's documentation relevant to your task."
        ),
        frequency="low_frequency",
        urls=(
            "https://github.com/acerbilab/pybads/blob/main/skills/pybads/"
            "SKILL.md",
        ),
    ),
)
