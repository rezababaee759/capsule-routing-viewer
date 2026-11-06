"""
Single source of truth for every protocol constant (see PROTOCOL.md).

Code imports these names; the manuscript receives them through
scripts/make_numbers.py, so no protocol number is typed anywhere else.
Lives in the explorer package so the app and the experiments share it.
Each constant names where its value comes from.
"""

# ---------------------------------------------------------------- sampling (PROTOCOL.md, "Image sets")
SEED = 2026                  # main-set / bootstrap / randomization seed
N_PER_CLASS = 50             # main set: images per class from the test split
MISCLASSIFIED_MAX = 100      # misclassified set: at most this many test errors per dataset
MISCLASSIFIED_SEED_OFFSET = 1
PILOT_SEED = 0               # pilot set, excluded from all sets
PILOT_N = 30

# ---------------------------------------------------------------- metrics (PROTOCOL.md, "Metrics")
TOPK_FRACS = (0.01, 0.05)    # top-k fractions for P@k; the first is the primary metric
TOP_SET_FRAC = 0.05          # top-set Spearman and magnitude metrics use the true top set of this size
FAR_FACTOR = 1.5             # a per-capsule estimate this far off Δ (either way), or of the wrong sign, counts as far
BUDGETS = (0.01, 0.02, 0.05, 0.10)   # group-deletion budgets
CURVE_STEPS = 40             # log-spaced deletion curve resolution

# ---------------------------------------------------------------- statistics (PROTOCOL.md, "Statistics")
N_BOOT = 5000
CI_LEVEL = 0.95
BCA_SKEW_THRESHOLD = 1.0     # BCa intervals when |skewness| exceeds this
ALPHA = 0.05                 # Holm family-wise level for the confirmatory tests
CONFIRMATORY = (("H1", "taylor_frozen"), ("H2", "coupling"))
PRIMARY_METRIC = "p1"

# ---------------------------------------------------------------- models / datasets
DATASETS = ("mnist", "fashionmnist", "cifar10")
SEED_VARIANTS = ("seed42", "seed123", "seed456")      # three training runs per dataset
VARIANTS = SEED_VARIANTS
# single-model analyses (tests, misclassified set, additivity, gates, randomization, cost, app):
# the run with the highest test accuracy on each dataset
REFERENCE_VARIANT = {"mnist": "seed123", "fashionmnist": "seed123", "cifar10": "seed42"}

# ---------------------------------------------------------------- randomization check
RANDOMIZATION_PER_CLASS = 10

# ---------------------------------------------------------------- cost benchmark
COST_WARMUP_GPU, COST_REPS_GPU, COST_REPS_EXACT_GPU = 20, 200, 50
COST_WARMUP_CPU, COST_REPS_CPU, COST_REPS_EXACT_CPU = 3, 20, 3
# Response-time limit for interaction that "feels instantaneous" (Nielsen, Usability
# Engineering, 1993, ch. 5) -- used only to shade the interactive zone in fig_cost.
INTERACTIVE_MS = 100


# ---------------------------------------------------------------- gate analysis (experiments/run_gates.py)
GATE_N_PER_DATASET = 10
GATE_PICK_SEED = 7
GATE_NOISE_SIGMAS = (0.1, 0.3, 0.5, 1.0, 0.02)   # each image draws its noise in this order
GATE_STEPS = 41        # gate α on a uniform grid over [0, 1]

# ---------------------------------------------------------------- group deletion against additivity (PROTOCOL.md)
ADDITIVITY_CALIBRATION_PER_CLASS = 1     # float32-vs-float64 calibration: the first main-set image of each class
ADDITIVITY_TOL_QUANTILE = 0.999
ADDITIVITY_TOL_FACTOR = 10               # τ, τ_c = factor × that quantile of |float32 − float64|
ADDITIVITY_TOL_SENSITIVITY = (0.1, 1.0, 10.0)

# ---------------------------------------------------------------- figure data (scripts/figure_data.py)
FIG_CURVES_PER_CLASS = 10
FIG_MAGNITUDE_PER_CLASS = 5

# ---------------------------------------------------------------- methods (PROTOCOL.md, "Reference and methods")
GRADIENT_METHODS = ("taylor", "taylor_frozen", "g_A", "g_C")
BASELINE_METHODS = ("activation", "coupling", "act_x_coupling", "random")
METHODS = GRADIENT_METHODS + BASELINE_METHODS
MAGNITUDE_METHODS = ("taylor", "taylor_frozen")   # methods that estimate Δ itself
SANITY_METHODS = ("exact", "taylor", "taylor_frozen", "g_A", "g_C", "activation", "coupling")
# (random is excluded from the randomization check: its map does not depend on the model.)
DELETION_FIGURE_METHODS = ("exact", "taylor", "taylor_frozen", "g_C", "g_A", "activation")

# ---------------------------------------------------------------- numerical tolerances
TIE_EPS = 1e-9          # consensus: coupling to ĵ must exceed every other class by more than this
SOFTMAX_TOL = 1e-3      # app sanity check: |Σ_j c_ij − 1| below this counts as a valid softmax
STORED_CURVES = 20      # full deletion curves kept per run (first images), for inspection

# ---------------------------------------------------------------- app display
ROUTE_DRAW_MIN = 0.01   # the 3D scene draws a route only if its coupling exceeds this (display choice)
