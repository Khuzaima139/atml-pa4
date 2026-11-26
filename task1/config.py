"""Fixed Task 1 settings: model, eligible projections, condition registry, batch sizes."""
import json

from common.paths import SOURCE_REVISIONS_LOCK

MODEL_ID = "Qwen/Qwen2.5-1.5B"


def _locked_revision(model_id: str) -> str:
    with open(SOURCE_REVISIONS_LOCK, encoding="utf-8") as fh:
        lock = json.load(fh)
    assert lock["repos"]["model_base"] == model_id, lock["repos"]["model_base"]
    rev = lock["revisions"]["model_base"]
    assert isinstance(rev, str) and len(rev) == 40, rev
    return rev


MODEL_REVISION = _locked_revision(MODEL_ID)
DTYPE = "float16"
ATTN_IMPLEMENTATION = "sdpa"

ELIGIBLE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")

SEED = 6304
GROUP_SIZE = 128
SPARSITIES = (0.25, 0.50, 0.75)
BITS = (8, 4, 3)
Q3_BITS = (2, 3, 4, 8)

# Evaluation identifier written into every Task 1 output.
EVAL_ID = "t1-eval-v1"

# Fixed batch sizes, one per benchmark, used for every condition.
PPL_BATCH_SIZE = 8
HELLASWAG_BATCH_SIZE = 32
CALIB_BATCH_SIZE = 8
LMEVAL_BATCH_SIZE = 32
# Batch size of both scorers (task1.evaluate and HFLM) in the lm_eval check run on the float32 model copy.
LMEVAL_FP32_BATCH_SIZE = 8
# The acceptance check is the lm_eval run on a float32 copy of the loaded model (task1.evaluate and HFLM with an FP32
# log-softmax, both on the copy). The two runs on the FP16 model (lm_eval default log-softmax in the model dtype, and
# FP32 log-softmax) are reported for information: there the two scorers pad and mask differently, so FP16 attention
# kernels can differ between them.
LMEVAL_ACCEPTANCE_RUN = "fp32_model"
# Positions per FP32 log-softmax chunk inside one sequence.
LOGPROB_CHUNK = 128

# Latency protocol.
LATENCY_WARMUP = 3
LATENCY_REPS = 10
DECODE_STEPS = 64
PREFILL_LENGTHS = (512, 2048)

# Benchmark sizes from the locked files.
PPL_SCORED_TOKENS = 50_000
HELLASWAG_FULL_N = 2_000
HELLASWAG_SUBSET_N = 512
N_CALIBRATION = 128
N_DEVELOPMENT = 32
CALIB_SEQ_LEN = 512

# Expected values for the real checkpoint (eligible counts include no biases).
N_LAYERS = 28
EXPECTED_ELIGIBLE_MATRICES = N_LAYERS * len(ELIGIBLE)
EXPECTED_ELIGIBLE_PARAMS = 1_310_195_712
EXPECTED_TOTAL_PARAMS = 1_543_714_304

HELLASWAG_FULL = "full"
HELLASWAG_SUBSET = "subset512"

# Condition registry. HellaSwag scope follows manual lines 164-166: full 2,000 IDs for the
# FP16 baseline, the three 50%-sparsity pruning methods and all INT4/INT3 quantization
# variants; the first 512 IDs for every other condition.
_FULL = HELLASWAG_FULL
_SUB = HELLASWAG_SUBSET
CONDITIONS = {
    "fp16": {"family": "baseline", "hellaswag": _FULL},
    "p1_s25": {"family": "P1", "sparsity": 0.25, "hellaswag": _SUB},
    "p1_s50": {"family": "P1", "sparsity": 0.50, "hellaswag": _FULL},
    "p1_s75": {"family": "P1", "sparsity": 0.75, "hellaswag": _SUB},
    "p2_s25": {"family": "P2", "sparsity": 0.25, "hellaswag": _SUB},
    "p2_s50": {"family": "P2", "sparsity": 0.50, "hellaswag": _FULL},
    "p2_s75": {"family": "P2", "sparsity": 0.75, "hellaswag": _SUB},
    "p3_s50": {"family": "P3", "sparsity": 0.50, "hellaswag": _FULL},
    "p3_s75": {"family": "P3", "sparsity": 0.75, "hellaswag": _SUB},
    "p2_24_s50": {"family": "P2_2of4", "sparsity": 0.50, "hellaswag": _SUB},
    "q1_b8": {"family": "Q1", "bits": 8, "hellaswag": _SUB},
    "q1_b4": {"family": "Q1", "bits": 4, "hellaswag": _FULL},
    "q1_b3": {"family": "Q1", "bits": 3, "hellaswag": _FULL},
    "q2_b8": {"family": "Q2", "bits": 8, "hellaswag": _SUB},
    "q2_b4": {"family": "Q2", "bits": 4, "hellaswag": _FULL},
    "q2_b3": {"family": "Q2", "bits": 3, "hellaswag": _FULL},
    "q3_int4": {"family": "Q3", "budget": "q2_b4", "hellaswag": _FULL},
    "q3_int3": {"family": "Q3", "budget": "q2_b3", "hellaswag": _FULL},
}


def hellaswag_n(scope: str) -> int:
    return {HELLASWAG_FULL: HELLASWAG_FULL_N, HELLASWAG_SUBSET: HELLASWAG_SUBSET_N}[scope]


def expected_shapes(config) -> dict:
    """[out, in] shape of every eligible projection, derived from a Qwen2 config."""
    hidden = config.hidden_size
    head_dim = getattr(config, "head_dim", None) or hidden // config.num_attention_heads
    q_out = config.num_attention_heads * head_dim
    kv_out = config.num_key_value_heads * head_dim
    inter = config.intermediate_size
    return {
        "q_proj": (q_out, hidden), "k_proj": (kv_out, hidden), "v_proj": (kv_out, hidden),
        "o_proj": (hidden, q_out), "gate_proj": (inter, hidden), "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }


# Shapes of the real checkpoint (scratch facts: hidden 1536, 12/2 heads of 128, MLP 8960).
FULL_MODEL_SHAPES = {
    "q_proj": (1536, 1536), "k_proj": (256, 1536), "v_proj": (256, 1536), "o_proj": (1536, 1536),
    "gate_proj": (8960, 1536), "up_proj": (8960, 1536), "down_proj": (1536, 8960),
}
FULL_MODEL_NON_ELIGIBLE_PARAMS = EXPECTED_TOTAL_PARAMS - EXPECTED_ELIGIBLE_PARAMS


def full_model_matrix_shapes() -> list:
    """Eligible [out, in] shapes of the real checkpoint in enumeration order (layer, projection)."""
    return [FULL_MODEL_SHAPES[n] for _ in range(N_LAYERS) for n in ELIGIBLE]


def settings_dict() -> dict:
    """Every fixed setting, for the run JSON."""
    return {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "dtype": DTYPE,
        "attn_implementation": ATTN_IMPLEMENTATION, "eligible": list(ELIGIBLE), "seed": SEED,
        "group_size": GROUP_SIZE, "sparsities": list(SPARSITIES), "bits": list(BITS),
        "q3_bits": list(Q3_BITS), "eval_id": EVAL_ID,
        "batch_sizes": {"perplexity": PPL_BATCH_SIZE, "hellaswag": HELLASWAG_BATCH_SIZE,
                        "calib_nll": CALIB_BATCH_SIZE, "lmeval_verify": LMEVAL_BATCH_SIZE},
        "lmeval_acceptance_run": LMEVAL_ACCEPTANCE_RUN,
        "logprob_chunk": LOGPROB_CHUNK,
        "latency": {"warmup": LATENCY_WARMUP, "reps": LATENCY_REPS, "decode_steps": DECODE_STEPS,
                    "prefill_lengths": list(PREFILL_LENGTHS)},
    }


# ---------------------------------------------------------------------------------------------------------------
# Nonuniform allocation (P3, Q3): sensitivity probes, allocation rules, candidate grid and selection criterion.
# Allocation unit: one full Transformer block; every eligible projection of a block receives the block's sparsity
# (P3) or bit width (Q3).
ALLOC_UNIT = "block"

# Pruning sensitivity measures (calibration file only).
#   S1: OWL outlier ratio of a block = fraction of the Wanda scores |W_ij| * ||X_j|| (dense calibration norms) of its
#       seven projections that exceed OWL_M times the mean Wanda score over those seven projections.
#   S2: isolated loss increase = calibration NLL with only that block Wanda-pruned at S (dense calibration norms,
#       other blocks dense) minus the FP16 calibration NLL, for S in PROBE_SPARSITIES.
OWL_M = 5.0
PROBE_SPARSITIES = (0.50, 0.75)
P3_MEASURES = ("S1", "S2")
# Quantization sensitivity measures (calibration file only), for b in PROBE_BITS.
#   QS1: isolated loss increase = calibration NLL with only that block AWQ-folded (scales searched at b) and quantized
#        at b, minus the FP16 calibration NLL.
#   QS2: sum over the block's seven projections of ||(W_eff - W) X||_F^2 / ||W X||_F^2 at b on dense calibration
#        inputs, W_eff in the original basis.
PROBE_BITS = Q3_BITS
Q3_MEASURES = ("QS1", "QS2")

# P3 rule: u = min-max normalised sensitivity over blocks (all equal -> 0.5); s_l = S + lambda - 2 lambda u_l;
# one shift c by bisection with mean_l clip(s_l + c, P3_S_MIN, P3_S_MAX) = S; per-matrix zeros = floor(s_l * numel),
# then single weights by largest remainder (ties: lower block sensitivity, then lower block, then projection order)
# up to the P2 total at S.
P3_S_MIN = 0.10
P3_S_MAX = 0.85
P3_LAMBDAS = (0.04, 0.08, 0.12, 0.16, 0.20)
P3_REFERENCE_LAMBDA = 0.0          # reference only, never selected; equals P2 per matrix
P3_BISECT_MAX_ITERS = 200
P3_BISECT_TOL = 1e-12              # |mean clip(s + c) - S| after bisection
P3_TARGETS = {
    "s50": {"condition": "p3_s50", "reference": "p2_s50", "sparsity": 0.50},
    "s75": {"condition": "p3_s75", "reference": "p2_s75", "sparsity": 0.75},
}

# Q3 rule: exact multiple-choice knapsack over blocks, bits in Q3_BITS, minimising sum_l sensitivity(l, b_l) (exact
# rational sums) subject to eligible packed bytes (task1.storage) <= the Q2 uniform budget; exact dynamic programming
# over integer byte totals; ties: fewer bytes, then the lexicographically smaller bit vector (block order).
# Non-eligible parameters are stored at FP16 and are identical across every Q3 candidate and its Q2 budget.
Q3_TARGETS = {
    "int4": {"condition": "q3_int4", "reference": "q2_b4", "bits": 4},
    "int3": {"condition": "q3_int3", "reference": "q2_b3", "bits": 3},
}
# Eligible packed bytes of uniform Q2 on the real checkpoint (task1.storage, group 128, FP16 scales).
Q3_FULL_MODEL_BUDGET_BYTES = {"int4": 680_687_616, "int3": 515_633_664}

# Selection on the development file (fixed): lowest development mean NLL among non-reference candidates whose
# allocation differs from the reference (P2 per-matrix zero counts, uniform Q2 bits); ties go to the smaller lambda,
# then measure order (S1 before S2, QS1 before QS2). References are recorded, never selected. Only if every
# non-reference candidate equals the reference is the same rule applied to them, and the winner is flagged
# collapsed_to_uniform in selected_allocation.json and the run JSON.
SELECTION_CRITERION = (
    "lowest mean NLL on the C4 development sequences among non-reference candidates whose allocation differs from "
    "the reference (P2 per-matrix zeros, uniform Q2 bits); ties: smaller lambda, then measure order S1 before S2 "
    "(QS1 before QS2); references (P3 lambda 0, Q3 uniform) are never selected; only if every non-reference "
    "candidate equals the reference is the collapsed allocation used, flagged collapsed_to_uniform"
)


def alloc_settings() -> dict:
    """Every fixed allocation setting, for run JSONs and selected allocation files."""
    return {
        "unit": ALLOC_UNIT, "owl_m": OWL_M, "probe_sparsities": list(PROBE_SPARSITIES),
        "probe_bits": list(PROBE_BITS), "p3_measures": list(P3_MEASURES), "q3_measures": list(Q3_MEASURES),
        "p3_s_min": P3_S_MIN, "p3_s_max": P3_S_MAX, "p3_lambdas": list(P3_LAMBDAS),
        "p3_reference_lambda": P3_REFERENCE_LAMBDA, "p3_bisect_max_iters": P3_BISECT_MAX_ITERS,
        "p3_bisect_tol": P3_BISECT_TOL, "p3_targets": P3_TARGETS, "q3_targets": Q3_TARGETS,
        "q3_full_model_budget_bytes": Q3_FULL_MODEL_BUDGET_BYTES, "selection_criterion": SELECTION_CRITERION,
    }


# ---------------------------------------------------------------------------------------------------------------
# Q3 design v2 (fold-group granularity). The block-level Q3 design above and its run-1 outputs are kept unchanged
# as an unsuccessful design (every candidate collapsed to the uniform Q2 allocation).
# Allocation unit: one AWQ fold group of one block; per block, in this order:
#   attn_in  (q, k, v; scale group qkv), attn_out (o; group o), mlp_in (gate, up; group gateup), mlp_out (down; group
#   down). A unit's bit width applies to all its projections and uses the calib_stats AWQ scales searched for its
#   group at that bit width; the four folds of every block are applied as in Q2.
Q3V2_DESIGN = "fold_group_v2"
Q3V2_UNIT = "fold_group"
Q3V2_UNITS = (
    ("attn_in", "qkv", ("q_proj", "k_proj", "v_proj")),
    ("attn_out", "o", ("o_proj",)),
    ("mlp_in", "gateup", ("gate_proj", "up_proj")),
    ("mlp_out", "down", ("down_proj",)),
)
Q3V2_BITS = Q3_BITS
# Quantization sensitivity measures v2 (calibration file only), per (unit, b), b in Q3V2_BITS:
#   QS1v2: restore(); apply only that unit's fold (its scale group, searched at b); quantize only that unit's
#          projections at b; calibration mean NLL minus the FP16 calibration mean NLL (same calibration sequences).
#   QS2v2: sum over the unit's projections of ||(W_eff - W) X||_F^2 / ||W X||_F^2 at b on dense calibration inputs,
#          W_eff = Q(W diag(s)) diag(1/s) with s the unit's own group scales at b (original basis).
Q3V2_MEASURES = ("QS1v2", "QS2v2")
# Exact knapsack: unit costs in chunks of g quantization groups, g = gcd of the units' group counts, and per-group
# bytes in eighth-bytes (codes group_size*b, FP16 scale 16, zero-point b), so every cost is an integer number of
# g/8-byte units; the budget is floor(8 * budget_bytes / g) such units.
Q3V2_FULL_MODEL_N_UNITS = N_LAYERS * len(Q3V2_UNITS)          # 112
Q3V2_FULL_MODEL_CHUNK_GROUPS = 3_072
Q3V2_FULL_MODEL_UNIT_CHUNKS = {"attn_in": 8, "attn_out": 6, "mlp_in": 70, "mlp_out": 35}
# Largest accepted difference between the FP16 calibration mean NLL recorded by the probe parts.
Q3V2_FP16_REF_TOL = 1e-6
Q3V2_RULE = (
    "exact multiple-choice knapsack over fold-group units (attn_in, attn_out, mlp_in, mlp_out per block), bits in "
    "{2,3,4,8}: minimise sum_u sensitivity(u, b_u) (exact rational sums) subject to eligible packed bytes "
    "(task1.storage) <= the Q2 uniform budget; costs in chunks of g groups (g = gcd of unit group counts) times "
    "eighth-bytes per group; dynamic programming over integer costs; ties: fewer bytes, then lexicographically "
    "smaller bit vector by unit order"
)
Q3V2_SELECTION_CRITERION = (
    "lowest mean NLL on the C4 development sequences among non-reference candidates (QS1v2, QS2v2) whose allocation "
    "differs from the uniform Q2 bits; ties: measure order QS1v2 before QS2v2; the uniform reference is never "
    "selected; only if every non-reference candidate equals the reference is the collapsed allocation used, flagged "
    "collapsed_to_uniform"
)


def q3v2_settings() -> dict:
    """Every fixed setting of Q3 design v2, for run JSONs, selected allocation files and artifact manifests."""
    return {
        "design": Q3V2_DESIGN, "unit": Q3V2_UNIT,
        "units": [{"unit": u, "scale_group": g, "projections": list(p)} for u, g, p in Q3V2_UNITS],
        "bits": list(Q3V2_BITS), "measures": list(Q3V2_MEASURES), "rule": Q3V2_RULE,
        "q3_targets": Q3_TARGETS, "q3_full_model_budget_bytes": Q3_FULL_MODEL_BUDGET_BYTES,
        "full_model_n_units": Q3V2_FULL_MODEL_N_UNITS, "full_model_chunk_groups": Q3V2_FULL_MODEL_CHUNK_GROUPS,
        "full_model_unit_chunks": Q3V2_FULL_MODEL_UNIT_CHUNKS, "fp16_ref_tol": Q3V2_FP16_REF_TOL,
        "selection_criterion": Q3V2_SELECTION_CRITERION,
    }
