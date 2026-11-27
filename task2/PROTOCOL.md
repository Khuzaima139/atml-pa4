# Task 2 protocol

Fixed before any Task 2 result exists. Each rule names the manual phrase it rests on, quoted from
docs/ATML_PA4.txt by line number (manual = the PA4 handout). Later changes are made only by dated amendments
appended at the end of this file and committed before the runs they affect.

Terms. "distance" is the recorded `distances` field of the supplied MQAR records: query position minus source
position, where the source is the key position and the value sits at source + 1. "Development set" is the supplied
`mqar_development.jsonl.gz`; "test sets" are the supplied `mqar_test.jsonl.gz` cells.

## Part A: exact attention

A1. Inputs: B = 2, H = 4, d = 64, L in {256, 512, 1024, 2048, 4096, 8192}, FP16 and FP32, seed 6304. FP32 inputs are
    drawn once per L; FP16 inputs are cast from them. Manual line 364-365: "Use B = 2, H = 4, dh = 64, sequence
    lengths L in {256, 512, 1024, 2048, 4096, 8192}, and FP16 and FP32 as supported."
A2. Errors of outputs and gradients (dq, dk, dv under one fixed random upstream gradient) are measured against dense
    attention in FP32 on the FP32 inputs. Manual line 365-366: "Compare outputs and gradients against the dense FP32
    reference at feasible lengths".
A3. A dense condition is attempted only where its projected peak memory is below 14 GiB; otherwise a skipped row with
    the projection is written, and any out-of-memory event is written as a row. Manual line 366: "benchmark larger
    shapes without forcing an infeasible dense baseline"; line 464-465: "Note unsupported backends or out-of-memory
    conditions explicitly."
A4. Timing: 3 warmup and 10 timed repetitions per condition with CUDA events and synchronization; peak statistics are
    reset per condition. Manual line 366-367: "Run ten warmed-up repetitions per supported condition"; line 61:
    "warm up the relevant path, synchronize CUDA around timed operations".
A5. The SDPA backend is recorded from profiler kernel names of the default call, and each backend is also forced on
    its own; an unsupported forced backend is recorded with its error and never replaced. Manual line 367: "Record
    the actual SDPA backend instead of assuming a particular fused kernel is used."
A6. The tiled implementation uses autograd for its backward pass, stores per-tile intermediates, and is reported as a
    Python reference, not an optimized FlashAttention kernel. Manual line 460-461: "If a Python tiled backward uses
    autograd and stores intermediates, report that limitation; do not label it as an optimized FlashAttention kernel."

## Part B: associative recall

B1. Seeds 6304, 6305, 6306. Seed s draws training indices [(s - 6304) * 144000, (s - 6303) * 144000) of each
    (K, regime) training stream, so the same seed gives identical batches to every architecture; the seed also sets
    initialization. Manual line 373: "run seeds 6304, 6305, 6306 for every required trained architecture and
    intervention"; line 372: "only the attention operator changes".
B2. Loss: 64-way cross-entropy over logits 128-191 at query positions only (label = id - 128); accuracy = argmax over
    those 64 logits. Manual line 391-393: "apply cross-entropy against the associated value ID at the same position,
    masking every other position. This is query-position classification with causal context"; line 393: "The
    label-space chance baseline is 1/64".
B3. Model: untied 256-output head, biases on, LayerNorm with weight and bias, FP32; 462,336 trainable parameters for
    the dense baseline. Manual line 371-372: "Use identical projections, normalization, optimizer, and training
    budgets across attention methods".
B4. Checkpoint curves at updates 500, 1000, 1500, 2000, 3000, 4500 are evaluated on the development set; test sets
    are evaluated only at update 4500. Manual line 381: "Checkpoint evaluation Updates 500, 1,000, 1,500, 2,000,
    3,000, 4,500"; line 69: "Do not use held-out test outcomes to choose".
B5. Training runs on a Kaggle T4. The forward-only sweep uses B = 2, FP32, T in {256, 1024, 4096, 8192}. Manual line
    409-410: "a separate forward-only inference sweep at T in {256, 1024, 4096, 8192} with fixed random embeddings and
    matched B, H, d".
B6. "Learned" means final development accuracy >= 0.5 on queries with distance < 64 in the run's own training
    regime. The dense gate at a given K passes if all three seeds learned. Manual line 541-542: "Verify the dense gate
    and show training curves before interpreting the other methods."
B7. Heatmap: every test query is pooled by recorded distance into bins [0,32), [32,64), [64,128), [128,256); the
    per (test set, bin) values are also saved. Manual line 406-408: "preserve and use the known source position.
    Report accuracy for finer distance bins [0, 32), [32, 64), [64, 128), [128, 256)".
B8. A difference between conditions counts only when the min-max ranges over the three seeds do not overlap. Manual
    line 373 (three seeds per architecture).
B9. Far-regime accuracy is additionally split by (distance - 1) mod 4. Manual line 305-306: "Its coverage also
    depends on alignment between source and query positions."
B10. The B3 intervention and its development-selection rule are added by a later amendment before B3 training.
    Manual line 482-483: "State a prediction before evaluation, and report ... its development-selection rule."

## Part C: pretrained KV memory

C1. Compression order: full FP16 prefill of the whole prompt; question-suffix query attention computed for scoring;
    the pre-question context cache compressed; the question suffix re-run on the compressed cache with absolute
    position_ids and physical cache_position; then greedy decoding. Manual line 441: "Measure the INT4 budget at the
    point where the pre-question context cache is compressed"; line 490-491: "preserve the original absolute rotary
    position IDs of retained tokens and use a consistent physical cache length when appending new tokens."
C2. Question-aware score: FP32 attention probabilities from all question-suffix queries to context keys (causally
    masked softmax over all cached keys), averaged over layers, all query heads and suffix positions; one global
    ranking shared by all layers and heads; no pooling. Manual line 498-499: "extract query projections and compute
    masked FP32 attention against existing rotary-positioned cached keys for importance scores"; line 500: "Specify
    whether retention is shared across heads/layers or selected independently".
C3. Retained positions are stored as one int32 list counted in bytes; sink-plus-recent stores no position list
    (8 bytes for two counts). The hybrid uses the same global ranking. Manual line 443-444: "Include physical
    packing/padding, any separately stored indices, and the FP16 residual in each method's total."
C4. Evidence token = any token whose character range overlaps the gold span. Manual line 522: "fraction of marked
    evidence tokens retained".
C5. Extractor: first match of `(?<!\d)\d{5}(?!\d)` in the generated text; no match counts as incorrect and is
    flagged. Manual line 426-427: "use a fixed, deterministic answer extractor."
C6. Generation: do_sample False, repetition_penalty 1.0, temperature, top_p and top_k unset, max_new_tokens 32, stop
    at EOS, supplied chat template unchanged (default system message included). Manual line 413-414: "greedy
    generation for quality evaluation, and the same prompt format across cache policies."
C7. Reference-answer NLL: mean NLL of the gold code's digit tokens, teacher-forced directly after the assistant
    header. Manual line 521: "reference-answer NLL".
C8. Logit divergence: KL(FP16 || method) over the full vocabulary at the first answer position. Manual line 521-522:
    "post-prefill logit KL from FP16".
C9. No development retrieval prompts are used, because no retention parameter is tuned: sink 8 and recent 64 are fixed
    by the manual. Manual line 495-496: "Fix 8 sink tokens and 64 most-recent tokens as safeguards in both rules";
    line 428: "Keep final-test prompts distinct from any development prompts used to choose fixed retention
    parameters."

## Predictions fixed before results

Each prediction is followed by its rival outcome.

A-H1. FP32: dense, tiled and SDPA within 1e-5 max abs error of the FP32 dense reference; FP16: dense output error
    larger than tiled at L >= 2048. Rival: FP16 errors equal within a factor of 2.
A-H2. Forward peak allocated memory log-log slope from L 2048 to 8192: dense >= 1.8; tiled and SDPA <= 1.2. Rival: SDPA
    dispatches a quadratic path on T4 for some dtype.
A-H3. Tiled forward plus autograd backward peak memory >= 0.8x dense at L >= 2048. Rival: < 0.8x.
A-H4. SDPA fastest in every condition; Python tiled slower than dense at L <= 1024. Rival: tiled faster than dense at
    some L >= 2048.
B-H1. Dense mixed learned at K 4 and 8; mixed accuracy falls from K 4 to 16; near-far gap < 10 points. Rival: dense
    fails at K 16 or far deficit > 10 points.
B-H2. Dense with test-time W = 64 mask within 5 points of unmasked dense on the near test set. Rival: drop > 10 points.
B-H3. Window trained on mixed K 16 more than 5 points below window trained on near K 16 on queries with distance < 64.
    Rival: within 5.
B-H4. Window models: [64,128) accuracy more than 30 points below [0,64). Rival: within 10.
B-H5. Linear accuracy drop from K 4 to 16 larger than dense's, near-far gap < 10 points. Rival: drop not larger, or
    linear fails at every K.
B-H6. Local/dilated far accuracy more than 20 points above window far accuracy at K 4, and higher by more than 10
    points when (distance - 1) mod 4 = 0. Rival: no alignment difference.

## Construction facts

F1. W = 64 models cannot reach far queries: 2(W - 1) = 126 < 128 (manual line 404-405).
F2. Distance < 64 is reachable by every mechanism.
F3. The mixed regime's distance mix differs by K, so comparisons across K use distance bins.
F4. The direct dilated hop reaches the value token only when (distance - 1) mod 4 = 0.
F5. Both INT4 layouts have identical byte counts by construction.

## Amendments

AM1 (2026-11-27, applied before any Part A result existed). A2 reference for FP16 rows. For every FP16 row of
`python -m task2.part_a verify` (accuracy, extreme-score and masked-row rows), the primary reference is dense
attention in FP32 computed on the FP16 inputs cast to FP32, and for gradients on the FP16 upstream gradient cast to
FP32, so FP16 input rounding is not counted as attention error. The comparison against dense FP32 on the original
FP32 inputs is kept for FP16 rows as a second set of error columns prefixed `vs_fp32_inputs_`. FP32 rows are
unchanged. Each row records its reference in `reference_inputs`; the result version is `task2-part-a-v2`.
