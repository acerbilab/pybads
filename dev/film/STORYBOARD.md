# The PyBADS film: storyboard

The film line by line: the narration, what each shot shows, and where the
narration's claims come from. Shot K is `film.html?shot=K`;
[README.md](README.md) says how the page plays the shots, and
[NOTES.md](NOTES.md) why the film is as it is. The script was locked on
2026-10-03, and line 1.5 rewritten on 2026-10-05.

The captions below are the narration word for word, but for three things.
The captions of lines 4.2, 7.2 and 7.3 end in an asterisk, which calls a
footnote at the bottom right. The caption of line 1.5, too long for the
two lines of the lower bar, shows in two parts, the second from "so its
slope". And `narration.json` spells some words as the
voice should say them: "Pie-Bads" for PyBADS and "Bads" for BADS, and
"search" and "poll" in lower case, so that the voice does not spell them
out.

On the draft voice the scenes start and last:

| Scene | Lines | Shots | Starts (s) | Lasts (s) |
|---|---|---|---|---|
| 1. The problem | 1.1–1.5 | 1–5 | 0.0 | 27.4 |
| 2. Bayesian optimization | 2.1–2.4 | 6–9 | 27.4 | 23.0 |
| 3. Direct search | 3.1–3.4 | 10–13 | 50.5 | 20.2 |
| 4. BADS | 4.1–4.5 | 14–18 | 70.7 | 30.1 |
| 5. A real run | 5.1–5.7 | 19–25 | 100.7 | 40.0 |
| 6. Close | 6.1–6.2 | 26–27 | 140.7 | 8.6 |
| 7. The record | 7.1–7.4 | 28–31 | 149.3 | 28.7 |

## 1. The problem

- **1.1** (shot 1) "Which parameters of your model fit your data best?"
  The plane of the two parameters. A point glides across it: each position
  is one setting of the parameters.
- **1.2** (shot 2) "The ones that minimize the error, or make your data
  most probable." The point glides on, and a stem rises from it to that
  setting's error, labelled "error": the height of a landscape not shown
  yet. The stem grows and shrinks as the point moves. "Most probable" is
  maximum likelihood, said plainly.
- **1.3** (shot 3) "This is an optimization problem." The plane holds, a
  little brighter, and the point glides on with its error: the line names
  the problem that the picture already shows.
- **1.4** (shot 4) "Picture the error as a landscape: the lower the ground,
  the better the fit. You want the lowest point." The true landscape rises
  out of the floor, grey, through the stem, whose top it meets: a long
  diagonal valley, and a narrow trench that leaves it and runs down to the
  lowest point.
- **1.5** (shot 5) "But you can't see the landscape. For many computational
  models, the error is approximated numerically, or noisy, so its slope is
  no guide, and each evaluation can take seconds: they are black boxes."
  The landscape fades out, the floor dims, and two small landscapes
  appear above it, in the look of the tiles of line 7.1. On
  "approximated numerically" the first lights, a bowl under bumps,
  labelled "approximated numerically"; on "noisy" the second, a bowl seen
  through noise redrawn every tenth of a second, labelled "noisy". On
  "slope" each bowl's bottom shows as a white point, and a white arrow,
  labelled "slope", points the way the local slope goes down: on the
  bumps, away from the bottom, down a bump's flank; on the noise, wherever
  the noise of the moment sends it, four times a second. On "each
  evaluation" the two step back, and one evaluation drops slowly onto the
  start, at its height, with a line down to the floor; on "black boxes"
  the empty floor takes the label *black box*. *Sources:* the FAQ's
  list of the problems that PyBADS suits (`docsrc/source/faq.md`, also in
  `README.md`): a landscape that is "rough (nonsmooth), typically due to
  numerical approximations or noise", and an objective "at least
  moderately expensive to compute (e.g., more than 0.1 s per function
  evaluation)". A slope can be computed or estimated, but on such a
  landscape it reflects the approximation or the noise, not the direction
  of the valley. The line speaks of many models, not all, nor of the
  film's landscape, which is smooth; line 7.1 shows rugged and noisy
  ones.

## 2. Bayesian optimization: smart and fast, but brittle

An illustration, from `illus.js`: a plain Bayesian optimization on the same
landscape, which starts from the run's initial design and picks each point
by the lower confidence bound μ − 0.5σ (NOTES.md, "The illustrations").

- **2.1** (shot 6) "Bayesian optimization fits a surrogate to the points it
  has seen: a statistical model of the whole landscape." Five evaluations
  drop. The surrogate rises out of them as a green sheet: bright where it is
  sure, faint where it is not.
- **2.2** (shot 7) "It evaluates where the ground looks low, or might be.
  Then it updates the surrogate, and repeats." Labels name the two kinds of
  ground: "looks low" where the sheet dips near the start, "might be low"
  where it is faint, down the valley. The first try goes to the faint edge
  of the box and lands higher than predicted; the sheet reshapes around it.
  Five more tries follow quickly, then the seventh, picked where the sheet
  is low and sure, lands lower than any point before. *The tries:* the
  first at (−5, −4.875), predicted +1.17 ± 4.90, lands at +3.18; the
  seventh at (−3.19, −3.06), predicted +1.42 ± 0.30, lands at +1.13.
- **2.3** (shot 8) "When the surrogate is accurate, this is smart and
  fast." A quick time-lapse: each point lands lower down the valley, and the
  sheet follows the valley's shape. *Source:* after 15 steps the
  illustration is at the bottom of the valley, where f = 0.
- **2.4** (shot 9) "But the surrogate can be wrong. If the landscape has a
  shape it can't capture, it keeps pointing to the wrong places." The last
  tries pile up on the valley's floor, where the surrogate keeps pointing.
  The true landscape appears faintly under the sheet: the trench, of which
  the smooth surrogate has no trace. *Source:* after 16 steps the
  illustration's best value is −0.97, where the trench begins and is still
  shallow; the minimum is −32.7.

## 3. Direct search: slow but steady

An illustration, from `illus.js`: a plain coordinate direct search on the
same landscape, from the same start.

- **3.1** (shot 10) "Instead, direct search uses no surrogate." Back to the
  start. No sheet: only the evaluations, and the best point so far, ringed.
- **3.2** (shot 11) "From the best point so far, it tries a step in each
  direction, on a grid called the mesh." The mesh appears as purple studs
  around the best point. The four steps go out along the parameters, one
  after another; each lands on the ground it measures. Here, all four are
  higher.
- **3.3** (shot 12) "If a step finds lower ground, it moves there and takes
  bigger steps. If none does, it takes smaller ones." One step finds lower
  ground: the best point moves there and the mesh spreads to twice the
  spacing. The next round finds nothing, and the mesh tightens again.
- **3.4** (shot 13) "It can't be fooled, and it comes with convergence
  guarantees. But it is slow." A time-lapse of the whole run: a staircase
  of small steps down the valley, then along the trench to the lowest
  point. *Sources:* "slow": the illustration needs 96 evaluations, and 68
  to get within 1 of the minimum, where the run of scene 5 needs 35.
  "Convergence guarantees": the convergence analysis of mesh adaptive
  direct search (Audet & Dennis 2006).

## 4. BADS

A diagram on black: two cards, SEARCH (green) and POLL (purple), each a
header over a panel that plays its stage in the look of scenes 2 and 3. The
SEARCH panel replays the run's first round of searches, the POLL panel
three polls of the illustration of scene 3 (NOTES.md, "Rules for the
pictures").

- **4.1** (shot 14) "Bayesian Adaptive Direct Search, or BADS, combines the
  two." The landscape gives way to the two panels, side by side: on the
  left a surrogate around the best point, on the right the best point on
  its mesh. Under them, "Bayesian optimization" and "direct search".
- **4.2** (shot 15) "It alternates a SEARCH stage, Bayesian optimization
  around the best point, and a POLL stage, a round of direct search.\*" The
  SEARCH card lights: a try is picked on the surrogate near the best point,
  lands lower, and the ring moves there. Then the POLL card lights: the
  cross grows out of the ring, its steps are tried in turn, the third lands
  lower, the ring moves and the mesh doubles. The footnote appears and
  stays through the scene: "\*The SEARCH and POLL stages come from mesh
  adaptive direct search (MADS; Audet & Dennis 2006)." *Source:* BADS
  builds on MADS (Acerbi & Ma 2017). The footnote says "come from" because
  the split predates MADS: the surrogate management framework (Booker,
  Dennis, Frank, Serafini, Torczon & Trosset 1999) already paired a SEARCH
  on a surrogate with a POLL.
- **4.3** (shot 16) "As long as the SEARCH finds better points, BADS keeps
  searching. When it fails several times in a row, BADS polls." The SEARCH
  plays on: a miss, a try that lands lower, a miss. Its four marks read
  ● ✕ ● ✕, and the loop over its card lights: "a better point: search
  again". Then the arrow to POLL appears, "fails 4 times in a row", its
  four crosses fill in, and on "BADS polls" the POLL card lights and its
  cross grows. *Source:* with the default options a round of SEARCH is
  `search_n_try` = 4 tries for two parameters, and BADS polls after a round
  without a success (`pybads/bads/bads.py`; NOTES.md, decision 2).
- **4.4** (shot 17) "The two stages help each other." The arrow back from
  POLL to SEARCH appears: the cycle is closed. Meanwhile the POLL finishes
  its round: none of its four steps lands lower, and the mesh halves.
- **4.5** (shot 18) "When the surrogate fails, the POLL moves away from the
  region it can't model, and the points it collects help build a better
  surrogate for the next SEARCH." The SEARCH card dims and its crosses
  flash. The POLL plays another round: its first step lands lower, the ring
  moves away and the mesh doubles; under it, "moves on without the
  surrogate". Its points glow and travel along the arrow back to SEARCH,
  "its points: a better surrogate", and the SEARCH card lights again.
  *Source:* a POLL's points enter the
  surrogate at its next rebuild (`pybads/bads/bads.py`).

## 5. A real run

The run of `trace.js`: PyBADS with default options and `random_seed=25` on
the same landscape (NOTES.md, "The run").

- **5.1** (shot 19) "Here is a real PyBADS run, on the same landscape."
  Dark landscape, the start, and the stage indicator top left. Nothing lit
  yet.
- **5.2** (shot 20) "It starts with a few evaluations and fits the
  surrogate. The SEARCH heads downhill fast." The first evaluations drop.
  The surrogate rises: a smooth valley. Then the SEARCH lights up and the
  best point runs down the valley, one round of four tries at a time.
  *Source:* evaluations 1 to 6 are the initial design, 7 and 8 a POLL that
  fails (shown without its indicator, among "a few evaluations"), and the
  rounds 9 to 12, 13 to 16 and 17 to 20 each contain a success.
- **5.3** (shot 21) "But the trench is too narrow for the surrogate. From
  one deep point, it predicts deep ground where there is none, and four
  searches in a row fail." Evaluation 20 lands in the trench's mouth, deep.
  The surrogate caves in: it predicts deep ground across whole regions.
  Four tries follow its predictions, and each lands far above them (white
  dashes). Four crosses. *Source:* evaluation 20 is at −8.54; the
  surrogate fitted with it predicts −228, −50, −39 and −7 for evaluations
  21 to 24, which land at +35.9, +30.6, +3.1 and +0.28.
- **5.4** (shot 22) "So PyBADS polls. One step lands in the trench, far
  lower than any point before, and the mesh grows." The surrogate dims. The
  cross goes out from the best point; its steps are tried in turn, in the
  order the surrogate suggests. The step along the trench lands far lower.
  The best point moves there, and the mesh studs spread to twice the
  spacing. *Source:* the POLL tries −x₁ (+4.36), +x₂ (+4.60), +x₁ (−24.75,
  where the surrogate predicted +52.5) and −x₂ (+12.91); the best value had
  been −8.54. The best point moves 2 along the trench, and the mesh step
  grows from 2 to 4.
- **5.5** (shot 23) "The points from the POLL show the surrogate the shape
  of the trench." The POLL's points glow. The surrogate refits on them, and
  a groove opens along the trench. *Source:* the refit takes the
  surrogate's length scales from (3.90, 3.51) to (2.61, 0.75).
- **5.6** (shot 24) "Now the SEARCH is fast again: its next point lands
  near the bottom of the trench." One try, picked on the new surrogate,
  lands near the bottom: the best point jumps along the trench. *Source:*
  evaluation 29, at (3.50, −0.53), predicted −22.7 ± 6.2, lands at −31.36;
  the minimum is −32.72.
- **5.7** (shot 25) "When no step improves any more, the mesh shrinks, and
  PyBADS settles on the minimum." A time-lapse of the end: searches that
  improve little, polls that find nothing, the studs closing in around the
  best point, until PyBADS stops. *Source:* the run stops at evaluation 87,
  at −32.715224, the true minimum. Eight later searches still lower the
  best value a little; PyBADS counts them as incremental improvements, not
  successes (NOTES.md, decision 2).

## 6. Close

- **6.1** (shot 26) "PyBADS never saw the true landscape. Here it is." The
  camera pulls back over the surrogate. On "Here it is", the true landscape
  rises in grey under it. Along the trench, the two coincide. *Source:*
  PyBADS saw only its 87 evaluations.
- **6.2** (shot 27) "Smart and fast when the surrogate is right. Slow but
  steady when it's wrong." The surrogate goes. The best point's whole path
  draws itself, coloured by the stage that made each move: green for the
  SEARCH, purple for the POLL.

## 7. The record

- **7.1** (shot 28) "That landscape had two parameters. BADS works with up
  to twenty, on rugged or noisy landscapes." The landscape of line 6.2
  shrinks into the first of three tiles, labelled "two parameters"; on "up
  to twenty" the label dissolves into "up to ~20 parameters". On "rugged or
  noisy landscapes" two more tiles appear beside it: a bowl under many
  bumps, "rugged", and a bowl seen through noise redrawn every tenth of a
  second, "noisy". *Sources:* the list of the problems that PyBADS suits in
  the FAQ (`docsrc/source/faq.md`, also in `README.md`): a rough landscape,
  "typically due to numerical approximations or noise", and "up to about
  D = 20" parameters; Singh & Acerbi (2024). The scope is that of BADS,
  which PyBADS ports.
- **7.2** (shot 29) "On dozens of real model-fitting problems, it matched
  or beat sixteen other optimizers.\*" The tiles give way to six charts,
  the six studies of the BADS paper's benchmark: the fraction of runs
  solved, by evaluations for the three deterministic studies (top) and by
  error tolerance for the three noisy ones (bottom). The curves draw in
  turn, and each name of the legend at the top comes in as its curves
  start: the other optimizers faint in grey, then two of them named,
  CMA-ES in white and gradient-based in light blue, then plain Bayesian
  optimization in green, and BADS last, in amber, above them all in every
  chart. The finished charts hold for 2.5 s after the line. Footnote:
  "\*Redrawn from Acerbi & Ma (2017): 36 problems from six studies, 6 to
  13 parameters, 50 runs per optimizer and problem, default settings.
  Fraction of runs solved, by evaluations (top) or, for noisy problems, by
  error tolerance (bottom). CMA-ES and gradient-based (fmincon): the best
  of their variants in each study." *Sources:*
  Acerbi & Ma (2017), section 4 and Figs 2 and 3: "Besides BADS, we tested
  16 optimization algorithms", plain Bayesian optimization besides, and "In
  all problems, BADS consistently performs on par with or outperforms all
  other tested optimizers, even when accounting for its extra algorithmic
  cost." In the charts redrawn from the benchmark's data (`bench.js`),
  BADS has the highest mean curve in all six studies (NOTES.md, "The
  record"). The two named optimizers are drawn in each study by the best
  of their variants, by mean curve: CMA-ES with active covariance
  adaptation (in its version for noisy problems in the noisy studies) in
  all six, and MATLAB's gradient-based `fmincon` in five, all but
  combinatorial game playing (interior point in the three deterministic
  studies; active set, the only variant run, in two noisy ones).
  One of the two is the best of the others in every study: CMA-ES in
  causal inference and the three noisy studies, `fmincon` in Bayesian
  confidence and neuronal selectivity.
- **7.3** (shot 30) "Since then, it has been used in hundreds of studies,
  from brains and quantum computers to oil wells and wildfires.\*" The
  charts fade. Hundreds of points light up across the floor, one for each
  study that has used BADS. As the narration names them, four points glow
  and take labels: "brains", "quantum computers", "oil wells",
  "wildfires". Footnote: "\*Brains: Cao et al. 2019; Tajima et al. 2019.
  Quantum computers: Than et al. 2025. Oil wells: Feng et al. 2022.
  Wildfires: Nobel et al. 2020." *Sources:* "hundreds of studies" is the
  count, by BADS's authors, of the studies that have used it. For each
  field the footnote cites studies whose own text shows that they used
  BADS or PyBADS: Cao et al. 2019, *Neuron*, fitted models of multisensory
  causal inference with BADS; Tajima et al. 2019, *Nature Neuroscience*,
  tuned a model of multi-alternative decisions in simulation, with an
  optimization that its Methods credit to the BADS paper; Than et al.
  2025, *Nature Communications*, optimized with PyBADS a variational
  quantum algorithm run on an ion-trap quantum computer; Feng et al. 2022,
  *Petroleum Science*, optimized a well's production with BADS; Nobel et
  al. 2020, *Journal of Environmental Economics and Management*, estimated
  with BADS a model of the impact of wildfires on the recreational value
  of heathland.
- **7.4** (shot 31, the end card) "Join them." The field of studies dims
  and its labels go. On "Join them." one new point pops in among the
  others, white. Then the end card fades in over the field: "PyBADS", then
  its version, "1.5", popping in with a short green glow, then the rest of
  the card: "Bayesian Adaptive Direct Search", `pip install pybads`,
  acerbilab.org/model-fitting, the two papers with their titles (Singh &
  Acerbi 2024, *PyBADS: Fast and robust black-box optimization in Python*,
  JOSS; Acerbi & Ma 2017, *Practical Bayesian optimization for model
  fitting with Bayesian adaptive direct search*, NeurIPS), and in small
  capitals "Machine and Human Intelligence Group · University of Helsinki",
  "Research Council of Finland · ELLIS Institute Finland" and "Directed by
  Luigi Acerbi · Made with Claude Code · Voice: Kokoro (draft)". The
  voice's credit follows the final voice.
