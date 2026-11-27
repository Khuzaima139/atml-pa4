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

AM2 (2026-11-28, applied before any Part C code or result existed). Part C cache arithmetic, layouts, metrics,
analysis and predictions. Symbols: n_l layers, H_kv KV heads, d_h head dimension, all read from the model config;
C = 2 n_l H_kv d_h stored KV channels (14,336 for Qwen2.5-1.5B-Instruct, so one FP16 position costs 2C = 28,672
bytes); L = pre-question context length in tokens; S = question-suffix length; r = L mod 64.

AM2.1 Quantizer. Asymmetric min/max per 64-element group: scale = (max - min) / (2^b - 1) and offset = min, both stored
    as FP16; codes = clamp(round_half_even((x - offset) / scale), 0, 2^b - 1), computed in FP32 with the FP16-rounded
    scale and offset; reconstruction = offset + scale * code. A constant group (or one whose FP16 scale rounds to 0)
    stores scale 0 and codes 0 and reconstructs to the FP16 offset, exactly for FP16 inputs. There is no integer
    zero-point; the FP16 offset plays that role. Codes are packed into uint8 low bits first (INT4: element 2i in bits
    0-3, element 2i+1 in bits 4-7; INT2: element 4i+j in bits 2j to 2j+1). Manual line 432-433: "Each quantization
    group has 64 elements, with an FP16 scale and FP16 offset (four metadata bytes), plus packed integer payload;
    zero-point handling is fixed and reported separately"; line 330-331: "This layout asymmetry is distinct from using
    asymmetric integer quantization with a nonzero zero-point."
AM2.2 Layouts. Token-wise: groups of 64 consecutive head-dimension elements per (layer, KV head, token). Channel-wise K:
    groups of 64 consecutive tokens per (layer, KV head, channel). V is token-wise in both layouts. In both INT4
    layouts the trailing r context tokens stay FP16 for K and V. Appended tokens (question suffix, generated tokens,
    teacher-forced gold digits) join the FP16 tail; in the quantized layouts the tail is quantized as one group when it
    reaches 64 tokens, after the forward pass that completed it (inside that pass those tokens are used in FP16, as
    produced). Eviction and the hybrid keep appended tokens in FP16. The bytes of appended entries are reported as the
    shared decode-time component and are included in total live cache memory. Manual line 433-435: "only complete
    64-token groups are compressed: the trailing r = L mod 64 cached tokens remain FP16 for both K and V, including
    newly generated tokens until they form a complete group"; line 504-505: "apply the same FP16 trailing-token rule to
    both layouts"; line 506-507: "Quantization must include prompt-cache entries and be applied consistently to
    subsequently generated entries"; line 441-443: "The identical question suffix and subsequently generated FP16
    tokens add a shared decode-time memory component, which must be reported separately and included when reporting
    total live cache memory."
AM2.3 Budget. M4 per prompt from M_b(L) = C [floor(L/64) (64b/8 + 4) + 2 r] with b = 4, at the pre-question context
    length L. Sink-plus-recent: largest N with 2C N + 8 <= M4 (8 bytes for the two counts; 8 sink, the recent window
    fills the remaining N - 8). Question-aware: largest N with 2C N + 4 N <= M4 (8 sink, 64 recent, then the
    highest-ranked remaining positions; one int32 index list). Hybrid: F = L - 64 m FP16 positions (8 sink, 64 recent,
    then the highest-ranked remaining positions), with m the smallest group count such that
    C m (64*2/8 + 4) + 2C F + 4 F <= M4; the other 64 m positions, in position order, form INT2 groups (channel-wise K,
    token-wise V); one int32 index list for the FP16 positions only (the INT2 positions are their ordered complement).
    Every count is capped at L, and a prompt where the 72 safeguards (or F >= 72) do not fit raises an error instead of
    being dropped silently. Bytes counted from the actual packed tensors must equal these formulas. Manual line 444-446:
    "retain the largest legal number of FP16 positions whose actual bytes do not exceed the reference INT4 budget.
    Apply the same actual-byte accounting to the hybrid, including its low-bit groups, quantization metadata, position
    indices, and FP16 positions"; line 511: "The low-precision positions must form complete 64-element groups, with
    any incomplete residual stored in FP16."
AM2.4 Computation uses stored values. K and V given to SDPA are dequantized from the packed codes (or are the stored
    FP16 values), in original position order; the packed object is the stored form and its tensors are counted. Manual
    line 530-531: "Distinguish packed storage from allocated GPU memory when quantized values are dequantized before
    ordinary SDPA."
AM2.5 Scoring details. Query projections are captured with forward hooks on each layer's q_proj during the full FP16
    prefill (attention implementation unchanged), rotated with the model's own rotary embedding at the absolute suffix
    positions in the model dtype, then cast to FP32; keys are the cached (rotated) keys cast to FP32 and expanded to the
    query heads. The global ranking places the 72 safeguards first, then the remaining positions by descending score
    with ties broken by lower position. Manual line 498-499 (quoted in C2).
AM2.6 Metrics per prompt and method: exact match (extractor C5); exact match conditional on FP16-correct prompts;
    reference-answer NLL (C7); post-prefill KL (C8, at the first answer position, after the suffix re-run on the
    method's cache); teacher-forced KL = mean over the 5 gold-digit positions of KL(FP16 || method), both models
    teacher-forced from their own cache state right after the suffix re-run; K and V relative reconstruction error
    ||X - X_hat||_F / ||X||_F over the L context positions, per layer and mean over layers (INT4 layouts and hybrid;
    0 for FP16; not defined for eviction); fraction of evidence tokens retained (eviction and hybrid FP16 sets);
    bytes by component (payload, metadata, FP16 retained positions, FP16 residual, indices, decode-time appended
    entries) and realized fraction of FP16 bytes M_method / M16(L); phase times (full prefill, scoring, compression,
    suffix re-run, decode); peak allocated memory. Generation stops at EOS or after 32 new tokens; the last generated
    token is not fed back. Manual line 520-523: "Record exact-match retrieval, reference-answer NLL, post-prefill logit
    KL from FP16, fraction of marked evidence tokens retained (for eviction and the hybrid's FP16 selection), realized
    packed bytes, scoring/preprocessing time, prefill time, decode time, and peak allocated GPU memory"; line 505-506:
    "Measure reconstruction errors for K and V separately, teacher-forced logit divergence, and free-running answer
    accuracy"; line 550-551: "also report compressed accuracy conditional on reference correctness".
AM2.7 Bootstrap: paired over the 180 prompts, 10,000 resamples, seed 6304. Principal differences, for exact match and
    NLL: each method minus FP16; question-aware minus sink-plus-recent; channel-K minus token-wise; hybrid minus
    question-aware; hybrid minus INT4 channel-K. Cell-level results are descriptive only. Manual line 524: "Use paired
    bootstrap resampling of prompts, with at least 2,000 resamples, for principal accuracy differences"; line 528-529:
    "treat the ten prompts in each grid cell as descriptive diagnostics rather than separately powered comparisons."
AM2.8 C5 benchmark: real caches from the first 8 prompt IDs (sorted) of each length level, truncated to L = 2048 and
    L = 4096; batch 1 uses the first of them, batch 8 all 8. Configurations: INT4 token-wise, INT4 channel-K, hybrid.
    3 warmup and 10 CUDA-synchronized repetitions per condition; pack time, unpack-plus-dequantize time, serialized
    bytes, and decode-step latency with unpacking versus FP16 at matched shapes. Manual line 525-527: "Independently
    benchmark actual packed-cache encode and decode overhead at batch sizes 1 and 8 and both context lengths. Use at
    least ten warmed-up, CUDA-synchronized repetitions per condition".
AM2.9 Qualitative set: for each context length, the lowest-ID prompt in each of the start, middle and end cells with 8
    distractors (6 prompts). Manual line 566-567: "a small fixed set of aligned generations with their evidence spans,
    FP16 selections and token-selection masks."

Predictions fixed before Part C results (manual line 482: "State a prediction before evaluation"), each followed by
its rival outcome:

C-H1. Full FP16 cache exact match >= 0.9 in every length x location cell. Rival: lower for middle evidence at 4k.
C-H2. Question-aware eviction exceeds sink-plus-recent by > 30 points overall; sink-plus-recent succeeds mainly when
    evidence lies in its recent window. Rival: gap < 10 points because question attention spreads across same-format
    distractor records.
C-H3. Question-aware retains >= 0.9 of evidence tokens and stays within 5 points of FP16 exact match at 0, 2 and 8
    distractors. Rival: exact match falls > 10 points from 0 to 8 distractors.
C-H4. Channel-K has lower K reconstruction error and lower teacher-forced KL than token-wise; both within 5 points of
    FP16 exact match. Rival: token-wise equal or better on KL.
C-H5. The INT4 layouts match FP16 exact match but answer NLL rises with a 95% interval excluding 0. Rival: interval
    includes 0.
C-H6. Hybrid exact match within 5 points of question-aware eviction; hybrid teacher-forced KL above INT4 channel-K.
    Rival: hybrid more than 10 points below eviction.
C-H7. Per-step unpack-plus-dequantize adds >= 20% to FP16 decode-step time at batch 8 and L 4096. Rival: < 5%.

Part C construction facts:

F6. Sink-plus-recent evidence retention is fixed by positions before the model runs.
F7. The hybrid FP16 set is a prefix of the question-aware ranking (F <= N), so its evidence retention cannot exceed
    question-aware eviction's.
F8. The FP16 reference has zero KL by definition.
F9. Full-prompt prefill is identical across methods, so eviction cannot lower prefill peak memory (manual line 543:
    "Eviction after a full prefill does not reduce the peak memory consumed during that prefill").
F10. Dequantizing before SDPA does not lower allocated memory during attention.

AM3 (2026-11-28, applied before any B3 code or result existed). B3 intervention, development-selection rule and
prediction (fills B10). Manual line 478-480: "Choose one intervention motivated by the observed results ... or a
justified curriculum that changes the ratio of reachable and unreachable queries"; line 480-481: "Train it at K = 16
for all three required seeds, using the same 4,500 updates and within 5% of the baseline trainable parameter count.
Keep final-test sets unchanged"; line 482-483: "State a prediction before evaluation, and report the intervention's
extra persistent state, the number of attended positions or state dimensions, and its development-selection rule."

AM3.1 Intervention: a near-to-mixed curriculum on the local_dilated mechanism at K 16 (run name
    local_dilated_curriculum_K16_s<seed>). Batch u (1-based) of 32 holds n_near(u) = round(32 f(u)) sequences from
    the seed's near K16 training stream and the other 32 - n_near(u) from its mixed K16 stream, near rows first; each
    stream is consumed in index order from the start of the seed's B1 index range
    [(seed - 6304) * 144000, (seed - 6303) * 144000), so no index is used twice and both streams stay inside that range.
    f(u) = 1 for u <= 1000, f(u) = (2000 - u) / 1000 for 1000 < u < 2000, f(u) = 0 for u >= 2000. round is
    round-half-to-even on the exact rational value (no tie occurs at 4,500 updates). Over 4,500 updates the run consumes
    47,984 near and 96,016 mixed sequences. f(u), n_near(u) and the near and mixed index ranges of every update are
    recorded. Both streams pass the generator validate() and filler checks and the disjointness check against the
    committed held-out hash list before training.
AM3.2 Motivation, from the B1/B2 development curves only (own cell, development accuracy on distance < 64 at the
    checkpoints): every near-trained window and dense run (window near K 4, 8, 16; dense near K 16) reached >= 0.997 by
    update 1000 on all seeds; dense mixed K 16 crossed 0.5 between updates 1000 and 2000 (seed 6304 by 1500, seeds 6305
    and 6306 by 2000); window mixed K 16, local_dilated mixed K 16 and two of the three linear mixed K 16 runs (seeds
    6304 and 6306) stayed between 0.04 and 0.09 at every checkpoint, around the 1/K = 0.0625 plateau, with mean training
    loss over the last 200 updates of 3.88-3.89 (window), 3.21-3.27 (local_dilated) and 2.81 (linear) against
    ln 16 = 2.77 and ln 64 = 4.16. Reading: K 16 queries are learned from near-only batches but not from mixed batches
    by the restricted mechanisms, so the curriculum starts local_dilated where the restricted mechanisms learned and
    then moves to the mixed ratio of reachable and unreachable queries.
AM3.3 Development-selection rule: the intervention and its schedule were chosen from the B1/B2 development learning
    curves (AM3.2) and fixed in this amendment before any B3 run; no B3 setting is selected on any data, and no B3
    checkpoint is chosen (the update-4500 model is the B3 model).
AM3.4 Everything else as B1/B2: architecture and 462,336 trainable parameters (0% change), optimizer, learning-rate
    schedule, gradient clipping, 4,500 updates, seeds 6304, 6305, 6306, checkpoint evaluations on the development set
    at the B4 updates, test sets evaluated only at update 4500 and unchanged, learned flag by B6 on the mixed K 16 cell
    (the regime of the final batches and of the comparator). Extra persistent state: none. Attended positions: as
    local_dilated (two heads per block on the preceding 64 positions, two heads at offsets {0, 4, ..., 252}).
    Comparator: local_dilated mixed K 16 (B2 runs, same seeds).
AM3.5 Prediction B3-H: local_dilated_curriculum learns (B6) on at least 2 of 3 seeds; its mixed K 16 test accuracy is
    more than 30 points above the comparator's (seed means; B8 applies); its far K 16 test accuracy is above 0.5 (seed
    mean). Rival R1: it learns near queries but far K 16 accuracy stays near 1/K. Rival R2: it stays flat (development
    accuracy near 1/K at every checkpoint).

AM4 (2026-11-28, applied after the notebook 2 Part C results and before any diagnostic code or result existed).
Post hoc diagnostic of the int4_token result. Manual line 505-506: "Measure reconstruction errors for K and V
separately, teacher-forced logit divergence, and free-running answer accuracy"; line 328-330: "A
channel-wise-K/token-wise-V layout, motivated by KIVI, groups key values along the token axis separately for each
feature channel, while grouping values along the feature dimension separately for each token"; line 72-73: "Report all
specified conditions, including runs that fail to reach useful accuracy".

AM4.1 Status. A post hoc diagnostic, motivated by the observed int4_token result (exact match 0/180 in
    results/task2/part_c/eval); not a matched-budget condition and not part of the principal comparisons (AM2.7). No
    Part C method, setting or row is changed, selected or dropped because of it. Version identifier task2-kv-diag-v1;
    entry point python -m task2.kv_diag; output directory results/task2/part_c/diag/.
AM4.2 Prompts: the lowest-ID prompt of each of the 18 grid cells (task2.kv_run.first_per_cell, the same set as the
    validate run's full-forward checks).
AM4.3 Channel statistics, on each prompt's FP16 pre-question context cache (the cached rotated keys and the values at
    positions [0, L) after the full FP16 prefill, the tensors every method compresses), per layer and KV head, over the
    L context tokens. Per channel: the max and the median of |K| and of |V| over tokens (median of an even count: the
    mean of the two middle values); for K and for V, the ratio of the largest channel max to the median channel max.
    Per-channel relative reconstruction error ||x_c - x_hat_c||_2 / ||x_c||_2 over the L context tokens, with x_hat the
    dequantized context of the packed cache as evaluated (the trailing r tokens are FP16 and contribute zero error): K
    under INT4 token-wise and under INT4 channel-K, V under INT4 token-wise. Channels are split per layer and KV head
    into the top 1% by channel max |K| (ceil(0.01 d_h) channels, 2 of 128; ties go to the lower channel index) and the
    rest; the summary of each group is the mean of its per-channel errors. V uses the same split by channel max |V|.
    Reported per layer as the mean over prompts and KV heads, and overall as the mean over layers.
AM4.4 Generation controls, with the same pipeline, settings and metrics as eval (C1, C6, AM2.6: exact match,
    reference-answer NLL, post-prefill KL and teacher-forced KL, the KLs against the fp16 reference of the same run,
    which is run on every prompt): (a) k_int4_token: K INT4 token-wise, V FP16; (b) v_int4_token: K FP16, V INT4
    token-wise; (c) kv_int8_token: K and V INT8 token-wise; (d) int4_token as evaluated; (e) int4_channel_k as evaluated.
    Quantizer (AM2.1; INT8 uses 2^8 - 1 levels and stores one code per byte without packing), group size 64, FP16 scale
    and offset per group and the trailing rule (AM2.2: the trailing r context tokens stay FP16 for K and V, appended
    entries join the FP16 tail, and a complete 64-token tail group is quantized after the forward pass that completed
    it) as AM2; only the bit width or the quantized tensor differs. In (a) and (b) the tensor that is not quantized keeps
    its FP16 values at every position. (d) and (e) run through the evaluated code path unchanged. Bytes are recorded;
    no budget applies.
AM4.5 Predictions fixed before running, on the 18 prompts: (c) exact match within 5 points of fp16 and mean
    teacher-forced KL below 0.05; (a) fails like (d): exact match at most 0.1; (b) exact match within 5 points of fp16;
    K shows channel max/median ratios far above V's (operationalised: the mean K ratio over prompts, layers and KV heads
    is at least 3 times the mean V ratio); token-wise K error on the non-outlier channels is far above channel-K's
    (operationalised: the overall non-outlier mean at least 3 times as large). Rival: (c) also fails (its prediction
    does not hold), which would indicate a defect in the token-wise code path rather than an effect of the layout.
AM4.6 Consequence. If (c) fails, int4_token is treated as a defect: the notebook 2 int4_token rows are kept as
    evidence, the defect is fixed and int4_token is re-run in a later notebook.
