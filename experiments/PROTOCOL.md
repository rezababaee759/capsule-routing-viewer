# Evaluation protocol

The evaluation asks whether the attribution that the Routing Explorer computes live can
stand in for exact ablation, what it costs, and what it reveals about routing. Every
constant named below lives in `explorer/protocol.py`; code and manuscript read it from
there, the manuscript through `scripts/make_numbers.py`.

## Evaluation questions
- **Fidelity:** how well do gradient attributions computed through the unrolled routing
  recover exact single-capsule ablation, in ranking and in effect size?
- **Robustness:** does this hold across datasets and independently trained runs?
- **Collective removal:** what happens when the most important capsules are removed
  together, and does the sum of single effects describe it?
- **Cost:** what does each method cost per image, and does it support interactive use?

## Models
- Datasets: MNIST, Fashion-MNIST and CIFAR-10, all with dynamic routing and
  `ROUTING_ITERS` iterations.
- Per dataset, three training runs with the same configuration and different seeds
  (`SEED_VARIANTS`). Fidelity, effect sizes, group deletion, routing measures and the
  margin check are computed on every run and reported per run and as the range over runs.
- The **reference run** of each dataset is the run with the highest full-test-split
  accuracy (`REFERENCE_VARIANT`). It carries every analysis of a single model: the
  confirmatory tests, the misclassified set, group deletion against additivity, the gate
  analysis, the randomization check, the cost benchmark, the case studies and the app.

## Image sets (test split only)
- **Pilot:** `PILOT_N` test images (`np.random.default_rng(PILOT_SEED)`), excluded from
  every set.
- **Main set:** `N_PER_CLASS` images per class, drawn with `SEED` outside the pilot; the
  same images for every run.
- **Misclassified set:** up to `MISCLASSIFIED_MAX` test images that the reference run
  misclassifies, drawn with `SEED + MISCLASSIFIED_SEED_OFFSET` outside the main set and the
  pilot. Descriptive only.
- **Additivity set:** every remaining test image, in index order. The misclassified set
  excluded is the one recorded in `results/fidelity/<dataset>_<reference>_misclassified.json`,
  because recomputing it on another device can change the seeded draw.

## Reference and methods
- **Reference:** exact single-capsule ablation Δ_i = φ(u) − φ(u with u_i = 0), φ = ‖v_ĵ‖,
  with routing re-run and ĵ the model's prediction.
- **Gradient methods** (`GRADIENT_METHODS`): the Taylor estimate ⟨∂φ/∂u_i, u_i⟩ through the
  unrolled routing, the same with the final couplings held constant (routing frozen), and
  the sensitivities ‖∂φ/∂u_i‖ and |∂φ/∂c_iĵ|.
- **Baselines** (`BASELINE_METHODS`): activation ‖u_i‖, coupling c_iĵ, their product, random.
- The ranking by exact Δ is called the single-capsule ranking, not an oracle.

## Metrics (per image)
- **P@1%:** the share of the true top-⌈0.01N⌉ capsules (by Δ) in the method's top set;
  also P@5% (`TOPK_FRACS`). Ties in a method's scores are broken by capsule index.
- **Rank correlations:** Kendall's τ_b and Spearman's ρ over all capsules, and Spearman's ρ
  on the union of the true and estimated top-`TOP_SET_FRAC` sets.
- **Effect size** (`MAGNITUDE_METHODS`): over the true top-`TOP_SET_FRAC` set, the
  least-squares slope of Δ̂ on Δ through the origin and the relative error
  Σ|Δ̂ − Δ| / Σ|Δ|; pooled over images, the quantiles of Δ̂/Δ and the share off by more than
  `FAR_FACTOR` or of the wrong sign.
- **Group deletion:** φ/φ0 after removing a method's top capsules together with routing
  re-run, on a log-spaced grid (`CURVE_STEPS`) and at the `BUDGETS`. Per image: the minimum,
  its location, the rebound (largest value after the minimum minus the minimum) and the area
  on the log grid. The curve is not monotone, so it is decomposed rather than reduced to an
  area.
- **Routing measures:** normalized routing entropy per iteration, for all capsules and for
  the true top 1%; consensus, the share of capsules whose coupling to ĵ exceeds every other
  class's by more than `TIE_EPS`; ‖v_ĵ‖ per iteration and the change in the last iteration;
  concentration, the share of the total positive Δ held by the true top 1%; and the relative
  size of the routing term, ‖∂φ/∂u − ∂φ/∂u|_c‖ / ‖∂φ/∂u‖.
- **Breakdowns** by class, and Taylor P@1% by correct and incorrect prediction, are
  descriptive.

## Statistics
- Summaries per dataset and run: median, quartiles, 5th percentile and mean over images,
  each with a `CI_LEVEL` class-stratified bootstrap interval (`N_BOOT` resamples, seed
  `SEED`, common random numbers across measures). The interval is the percentile interval,
  or BCa when the absolute skewness of the per-image values exceeds `BCA_SKEW_THRESHOLD`,
  with the acceleration from an unstratified jackknife.
- Images are never pooled across runs; across runs, the per-run summaries and their range
  are reported.
- **Confirmatory tests** (`CONFIRMATORY`), on P@1% of the reference run: Taylor against the
  routing-frozen Taylor estimate (H1) and against the coupling (H2), on each dataset, six
  tests with Holm's correction at `ALPHA`. Each reports the paired mean difference with its
  bootstrap interval, a two-sided Wilcoxon signed-rank p-value with Pratt's treatment of
  zeros, the matched-pairs rank-biserial correlation under the same convention, and the
  number of images on which Taylor is better. A missing method stops `summarize.py`, so the
  family cannot shrink.
- No other p-values: every other comparison is a descriptive estimate with its interval.

## Randomization check (descriptive)
- Cascading re-initialization of the routing weights W_ij, then the primary-capsule
  convolution, then the first convolution (Gaussian with each tensor's original standard
  deviation, seed `SEED`), on the first `RANDOMIZATION_PER_CLASS` main-set images of each
  class of the reference run, for `SANITY_METHODS`.
- Reported: the median Spearman ρ between each method's attribution on the trained and on
  the randomized model, the agreement of the Taylor estimate with exact ablation on the
  randomized model, and the class each model predicts.

## Gate analysis
- `experiments/run_gates.py` rebuilds the network in double precision and, on
  `GATE_N_PER_DATASET` main-set images per dataset of the reference run (drawn with
  `GATE_PICK_SEED`), evaluates the gate f_i(α) = φ(u with u_i scaled by α), routing re-run
  and ĵ fixed, at `GATE_STEPS` values of α in [0, 1], for the capsules at true ranks 1, k,
  k+1 and ⌈`TOP_SET_FRAC`·N⌉, where k is the size of the true top 1%.
- Also recorded: the ratio of the smallest effect inside the true top 1% to the largest
  outside; Taylor/Δ around the cutoff and its difference between ranks k and k+1; the share
  of pairs across the cutoff, over true ranks ⌈k/2⌉ to k + ⌈k/2⌉, that Taylor orders
  correctly; the top-1% capsules Taylor misses; P@1% of the exact effects under multiplicative
  noise (`GATE_NOISE_SIGMAS`, one draw per image); and the log–log fit of Taylor against Δ on
  the true top `TOP_SET_FRAC`.

## Group deletion against additivity (reference run, additivity set)
- **Order and interventions:** capsules are removed in descending order of Δ (stable ties);
  S_k are the first k. For every k = 0..N: φ_R(S_k) with routing re-run, and φ_F(S_k) with the
  final couplings c* of the intact input held fixed, φ_F(S) = ‖squash(s_ĵ(S))‖ with
  s_ĵ(S) = Σ_{i∉S} c*_iĵ û_ĵ|i.
- **Single effects and interactions:** Δ_i (routing re-run) and Δ^F_i (couplings fixed);
  I_R(S) = [φ0 − φ_R(S)] − Σ_{i∈S} Δ_i and I_F(S) = [φ0 − φ_F(S)] − Σ_{i∈S} Δ^F_i, and
  R(S) = [φ_R(S) − φ_F(S)] / φ0. Since I_R − I_F = −R − Σ_{i∈S}(Δ_i − Δ^F_i)/φ0, the difference
  of the interactions includes the difference between the two kinds of single effect.
- **Markers:** k₊ = #{Δ > τ}; k₋, the first k whose removed set contains a capsule with
  Δ < −τ; and the first k at which Σ_{i∈S_k} Δ_i ≥ φ0.
- **Pre-negative rise:** the largest rise of φ_R/φ0 over any earlier point, both within
  k ≤ k₊. An additive curve cannot rise there, so a rise above τ_c contradicts additivity for
  that image. The same is computed for φ_F, over the leading part of the order in which every
  Δ^F > τ, and for φ_F in its own Δ^F order.
- **Rebound budget:** an additive curve can rise by at most Σ_{Δ<0} |Δ| / φ0; reported is the
  share of images whose rebound on the deletion grid exceeds it by more than τ_c.
- **Direction:** the cosine between s_ĵ(S_k) and s_ĵ(∅) under fixed couplings, and between the
  class capsule after and before removal with routing re-run, at the minimum of φ_R before
  k₊, together with whether the predicted class has changed there.
- **Values at fixed points:** I_R, I_F and R at the `BUDGETS`, at the minimum of φ_R before k₊
  and at k₊; minimum, location and rebound on the deletion grid for φ_R, φ_F and φ_F in its
  own order.
- **Tolerances:** on the first `ADDITIVITY_CALIBRATION_PER_CLASS` main-set image(s) of each
  class, single precision on the run device is compared with double precision on the CPU.
  τ is `ADDITIVITY_TOL_FACTOR` times the `ADDITIVITY_TOL_QUANTILE` quantile of |Δ32 − Δ64|,
  and τ_c the same for |φ_R32 − φ_R64| / φ0 on the deletion grid. The markers, the rises and
  the shares of images with a rise above τ_c are also computed with both tolerances scaled
  by each factor in `ADDITIVITY_TOL_SENSITIVITY`; the fixed points and the comparison with
  the rebound budget use the main tolerances.
- **Statistics:** descriptive only, per dataset, with the bootstrap of the rest of the study;
  shares of images are means of indicators. Correct and incorrect predictions are broken
  down separately.

## Choice of score (margin check)
On the main set of every run, each capsule is also scored by its effect on the margin
m = ‖v_ĵ‖ − max_{j≠ĵ} ‖v_j‖, with the runner-up re-chosen after each ablation; the Taylor
margin estimate uses the intact runner-up. Reported: Taylor-margin against exact margin
(P@1%, τ), the exact length and exact margin rankings against each other (top-1% overlap,
τ), Taylor-length against exact margin (P@1%), and the share of single ablations that change
the runner-up or the prediction.

## Cost (descriptive)
One image per dataset (the first of the main set), batch size 1, reference run. GPU: the
forward pass, the gradient attributions and exact ablation (batched), with
`COST_WARMUP_GPU` warm-up and `COST_REPS_GPU` timed runs (`COST_REPS_EXACT_GPU` for exact
ablation). CPU of the same machine: `COST_WARMUP_CPU` warm-up and `COST_REPS_CPU` timed runs
(`COST_REPS_EXACT_CPU` for exact ablation). Medians and quartiles in milliseconds; app
timings are single runs. Timings are not compared across machines.

## Figures and case studies
- Deletion curves use the first `FIG_CURVES_PER_CLASS` main-set images of each class; the
  magnitude scatter uses the first `FIG_MAGNITUDE_PER_CLASS`.
- Case studies are chosen by rule on the reference run: the correct main-set prediction with
  the largest φ, the correct one with the smallest margin between the two longest class
  capsules, and the misclassified-set image with the largest φ.
- The routing-scene screenshot is illustrative; WebGL rendering is not byte-reproducible.
  Every plotted value comes from a result file.

## Provenance and determinism
Every script enables deterministic kernels where PyTorch offers them. Every result file
records its command line, a fingerprint of the source code, library versions, device, seeds,
image indices, checkpoint md5s and the md5 of every input. Paper numbers are generated from
these files, never typed by hand.
