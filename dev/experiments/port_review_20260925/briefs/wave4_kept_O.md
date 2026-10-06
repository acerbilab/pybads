# Wave 4: the items kept from the O reviewer

The text that replaces `{KEPT_ITEMS}` in the O verifier's prompt
(`wave4_verifier.md`), from the records named in the plan's "Wave 4
pickup", step 4. Each item is quoted from its record; a note in brackets is
the orchestrator's and says what the records of the review add. Wave 0's
fix pass reached `dev-next` squash-merged as `0c56d86`, wave 1's as
`fef6c14`, wave 2's as `8aecb6a` and wave 3's as `0d866e8`; the commits of
the fix passes that the items cite are in the clone. The survey's candidate
table has no open row in this slice's code (its rows on the hedge's reward,
the length scale's accumulator and the target were closed by W3-6, W1-18
and W3-21), and none of the differences that the preparatory agent saw in
passing belongs to it. The items below were left, by the fix passes of
waves 1 and 3, to the slices that own their code (B3, B4, B5), whose waves
have passed; slice O reads the same code.

---

- **O-K1.** Wave 1's ledger, "Found while fixing", reported by fix agent B: "with several hyperparameter samples, the poll scale sums over them unweighted and the effective radius is one per sample, where MATLAB weights by `hypweight` and averages; inert with one sample". [PyBADS optimizes one hyperparameter set (the sheet's KD-B5-4); say whether any option or path gives the GP more than one sample at `0d866e8`, and so whether the item is reachable.]
- **O-K2.** Wave 3's ledger, "Found while fixing", reported by fix agent A: "`acq_fcn_lcb`'s summary line says that it retrieves a point, and it computes an unused `n`; `update_hedge`'s docstring speaks of a probability of improvement".
- **O-K3.** Wave 3's ledger, "Found while fixing", reported by fix agent A: "`hedge_gamma` is not checked (above `1/n` the hedge's probabilities invert, above `1/(n-1)` they turn negative)". [`n` is the number of search strategies, 2 at default; the default `hedge_gamma` is 0.125. Say what MATLAB's `searchHedge.m` does with such a value.]
- **O-K4.** Wave 3's ledger, "Found while fixing", reported by fix agent A: "`sqrt_beta` is checked at the first search, not when `BADS` is created, and the value that a callable returns is not checked". [W3-10 (`599115b`) made `acq_fcn_lcb` refuse a `sqrt_beta` that is not `None`, a callable or a positive finite number.]
- **O-K5.** Wave 3's ledger, "Found while fixing", reported by fix agent D: "Fig. 1 of the documentation (`docsrc/source/_static/bads-cartoon.png`, in `README.md` and `index.rst`) draws the poll's steps anisotropic, which `poll_scale` does not make them (W3-25)". [W3-25, "not a defect": the poll divides by `poll_scale` and multiplies it back, as MATLAB does; `poll_scale` shapes only the ES-ell search. Say what the figure shows and what the code does.]
