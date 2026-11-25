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
        "logprob_chunk": LOGPROB_CHUNK,
        "latency": {"warmup": LATENCY_WARMUP, "reps": LATENCY_REPS, "decode_steps": DECODE_STEPS,
                    "prefill_lengths": list(PREFILL_LENGTHS)},
    }
