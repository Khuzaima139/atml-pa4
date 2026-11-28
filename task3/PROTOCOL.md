# Task 3 protocol: draft-model distillation for speculative decoding

Fixed on 2026-11-28, before any Task 3 model output exists. Later rules are added as dated amendments (AM1, AM2, ...), each committed alone before the code that uses it. Sources: docs/ATML_PA4.txt (Task 3 section), data release external/atml-pa4-datasets at commit 6a0a9ec, model revisions in data_lock/local_source_revisions.json.

## 1. Settings fixed by the manual
- Target and teacher: Qwen2.5-1.5B-Instruct, FP16, frozen. Draft and student: Qwen2.5-0.5B-Instruct, FP16 frozen backbone, LoRA r=8, alpha=16, dropout 0 on q_proj, k_proj, v_proj, o_proj; FP32 adapters and optimizer states.
- Seeds 6304 and 6305; methods off-policy and on-policy; each run starts from the original student.
- Quota per run: 40,000 teacher-scored response tokens (30,000 mathematics, 10,000 short form). Caps: 192 mathematics, 32 short form. Temperature 1, top-p 1.
- AdamW, LR 3e-4, weight decay 0.01; FP16 autocast with dynamic gradient scaling; unscale before clipping; global norm 1.0; microbatch 1; one update per 256 scored tokens; KL normalised by supervised tokens per update.
- Loss: forward KL(p_teacher || q_student) over the full vocabulary at every supervised response prefix.
- Held-out: 120 prompts (60 GSM8K test, 60 short form: 30 familiar templates, 30 shifted templates). Gamma in {2, 4}, two paired repetitions. Speed benchmark cap 64, EOS-aware. Answer-quality caps 512 mathematics, 32 short form. Fixed-64 diagnostic on 24 mathematics prompts, EOS ignored for timing only.
- Sampler test: three-symbol Markov toy, length 3, 100,000 fixed-seed sequences, all expected counts >= 10, correct sampler not rejected at p < 0.01, residual-replaced-by-p sampler rejected at p < 0.01, joint-sequence probabilities shown.

## 2. Decisions
D1. Prefix banks (i) teacher and (ii) original_draft are produced by our own committed generator. The release script (external/atml-pa4-datasets/scripts/setup_data.py, --prefix-banks, lines 106-150) raises ValueError at generate(generator=...) under transformers 4.57.1, and it merges the models' generation_config (repetition_penalty 1.1), which is not the temperature-1, top-p-1 law of the manual and would make banks (i) and (iii) differ by construction. Our generator keeps everything else from the release script: the 120 held-out prompts (60 GSM8K test questions as GSM8K-{order:04d}, 60 short_test inputs), a single user message with the chat template, one generation per prompt and bank with at most 192 new tokens, seed 6304 + 10007*j (+0 teacher, +123 original_draft) for prompt index j, positions drawn with random.Random(f"PA4-banks-v1:6304:{pid}:{bank}") in bins (1,16,8), (17,32,8), (33,64,16), (65,192,16) as min(n, available) sorted positions, and position p storing the prefix seq[:p-1]. Recorded as a deviation.
D2. Bank (iii) exact_spec_original_gamma4 is produced by our exact speculative sampler with the original student at gamma 4, same prompts and position scheme as D1, seed 6304 + 10007*j + 456. It is generated only after the toy test, the negative control and the cache, bonus, EOS and cap tests pass and results/task3/sampler_check.json is committed, and after the real-model preflight passes.
D3. All sampling (teacher collection, on-policy rollouts, banks, speculative decoding, target-only reference, standalone evaluation) uses one shared function: fp32 softmax of logits / 1.0 over all 151,936 rows, no top-k, no top-p filtering, no repetition penalty, torch.multinomial with an explicit torch.Generator per sequence. model.generate is not used for sampling. Batch size 1 everywhere, so no padding exists.
D4. Input format: the chat template with add_generation_prompt=True and no explicit system message, so the template's default Qwen system message is present for every model and condition. Identical for teacher and student (token IDs verified identical on all 920 prompts).
D5. EOS: generation stops at the first token in {151645, 151643}. The EOS token is part of the response; the position that predicts it is supervised and scored. A response that hits its cap has exactly cap response tokens and termination reason "cap".
D6. Vocabulary: KL, TV, greedy agreement and sampling use all 151,936 lm_head rows of both models.
D7. Training schedule (both methods, per training seed s): each category (mathematics 500, short form 300) has its own permutation drawn with seed s, cycled in that order, reusing prompts after exhaustion. Each presentation takes the next prompt from the category with the smaller fraction of its quota filled (mathematics on ties). A category stops when its quota is met; the response that crosses a quota is scored only up to the quota and recorded as truncated. On-policy and off-policy runs with the same seed use the identical schedule.
D8. Off-policy collection: one frozen teacher collection shared by both seeds, generated before training with collection seed 6304: one response per mathematics prompt (K=1) and 16 per short-form prompt (K=16), response k of prompt i with a rollout seed derived deterministically from (6304, prompt ID, k). The n-th presentation of prompt i in a run uses response (i, n mod K); any wrap-around is recorded. Stored per response: source ID, k, rollout seed, response token IDs, termination reason, length.
D9. On-policy rollouts: generated by the current student (adapters active, no gradient, eval mode) when the presentation starts, temperature 1, top-p 1, caps 192 and 32, rollout seed derived deterministically from (training seed, presentation index). Stored as in D8 plus the index of the last completed update.
D10. Update rule: supervised positions are consumed in schedule order and every update contains exactly 256 scored positions, except the final update of 40,000 - 156 x 256 = 64 positions. A response whose positions cross an update boundary is split; its remaining positions are recomputed with the updated student in the next update. 157 updates per run. Skipped updates are detected from a drop in the gradient scale or a non-finite clip norm after unscaling and recorded with their index; they still consume their positions.
D11. Training data: all 500 GSM8K training IDs and all 300 short-form training examples, including the 4 duplicate inputs, used as given. The 3 shifted test templates never enter training.
D12. LoRA: peft==0.22.0 in the Mac and Kaggle requirements, torchao uninstalled on Kaggle, peft default initialisation (B = 0) under torch seed = training seed. The adapter after update 157 is the only checkpoint evaluated; no checkpoint selection.
D13. Fixed-64 diagnostic: the 60 GSM8K test IDs in manifest order, permuted with numpy default_rng(6304), the first 24 kept, saved as task3/fixed64_ids.json and committed before any held-out output exists.
D14. Own-student banks: one per trained student (4 banks), generated by that run's merged final draft with the D1 prompts and position scheme, seed 6304 + 10007*j + 1000 + r for runs r = 1..4 (off-policy 6304, off-policy 6305, on-policy 6304, on-policy 6305), producing checkpoint recorded. All five drafts (original and the 4 trained) are evaluated on all 7 banks with identical prefix IDs.
D15. Speed benchmark: per (prompt, repetition) one sampling seed shared by all drafts and the target-only run. Each speculative run has an adjacent target-only run on the same prompt and seed (immediately before in repetition 1, immediately after in repetition 2). Draft order is rotated by a 5 x 5 Latin square over prompts and recorded. 3 warm-up generations per model before timing. Merged drafts are primary; one unmerged adapter (off-policy seed 6304) is timed separately.
D16. Rounds near the cap: a round proposes k = min(gamma, r) tokens, r the remaining allowance; tokens beyond the cap or after EOS are discarded, and the bonus token is skipped when it would exceed the cap.
D17. Standalone accuracy: each of the five drafts generates with D3 sampling at caps 512 and 32, two repetitions, seeds shared across drafts. Mathematics extraction: the number after "####" if present, else the last number in the response, commas removed. Short form: exact match after stripping whitespace, plus per-template format compliance by a regex derived from each template's instruction. Extractor and regexes are committed with one scorer identifier, checked to accept 100% of gold answers, before any model output exists. Unextractable and truncated responses stay in every count.
D18. Uncertainty: paired percentile bootstrap over prompts, 10,000 resamples, seed 6304; repetitions averaged within a prompt first; each training seed reported separately.
D19. Qualitative selections, chosen by rule from committed outputs: (a) aligned prefix-level disagreements: in each workload group (mathematics, familiar, shifted), on bank (i), the 2 positions with the largest TV difference between on-policy and off-policy (seed 6304) and the 2 with the largest TV reduction from the original to off-policy seed 6304, with the top-5 tokens of each model; (b) matched continuations: for gamma 4, repetition 1, original draft, in each workload group the prompt with the median measured speedup and the prompt with the lowest, shown for all five drafts and the target-only run.
D20. Merged versus unmerged check: max abs logit difference on the first 4 mathematics and first 4 short-form prompts of the seed-6304 schedule, per run.

## 3. Metric definitions
- Proposed: draft tokens generated in a round (k). Examined: proposals that underwent the acceptance test (accepted + 1 if a rejection occurred). Accepted/examined and accepted/proposed are pooled over rounds per condition.
- Mean accepted prefix per verification: accepted proposals / verification passes.
- Target passes per output token: verification passes / output tokens, excluding prefill (reported separately).
- Draft forward passes: every draft forward call, including catch-up feeds after full acceptance.
- Per-round time decomposition (CUDA-synchronised): draft proposal, target verification, acceptance and residual sampling, cache rollback or advancement. Prefill and remaining host time reported separately.
- Isolated decode cost: one-token decode step of each model at context lengths 64, 128, 256 and 512, 5 warm-up and 20 timed repetitions; target verification passes over gamma + 1 tokens at the same lengths.
- c: draft step time / target step time (medians, matched context length). v_gamma: target verification-pass time / target step time. alpha_hat: accepted/examined per condition. S_hat from the manual's formula with these; break-even alpha solved numerically on [0, 1], or "none".
- Measured S_gamma: per-prompt target-only time / speculative time (paired), plus tokens per second.
- Bank metrics per position: forward KL, reverse KL, TV, expected acceptance 1 - TV, greedy agreement; token-weighted means with the same denominators for every draft.
- Consistency check (i) versus (iii): per prompt, mean target entropy over saved positions, mean original-draft 1 - TV, response length and EOS within 64; bootstrap over prompts of the (iii) minus (i) difference.
- Deployment break-even: total distillation cost (teacher collection, rollouts, teacher scoring and training, merging) divided by the measured per-request time saved, against the original draft and against target-only; "none" if no saving.

## 4. Construction facts (not findings)
- CF1. All runs have 157 updates by D10; any difference is skipped updates only.
- CF2. Expected acceptance equals 1 - TV by identity.
- CF3. Banks (i) and (iii) follow the same law; a systematic gap indicates an implementation or evaluation problem.
- CF4. Speculative outputs follow the target law for every draft; speculative output quality is never used to rank drafts.
- CF5. accepted <= examined <= proposed; target passes per output token >= 1/(gamma+1).
- CF6. Short-form responses are short: bins [33,64] and the 65-192 bank hold mostly or only mathematics. Every bin reports its n; empty cells stay empty.
- CF7. The 64-token cap truncates most mathematics responses (52 of 60 gold solutions are longer), so EOS effects in the speed benchmark come mainly from short form.
- CF8. With B = 0 at initialisation every student equals the original before update 1.
- CF9. Off-policy seeds share one collection; their differences come from schedule order and LoRA initialisation only.
- CF10. Resolution: 60 items per domain (1 item = 1.67 pp), 30 per short-form template group (3.33 pp).

## 5. Predictions with rivals
Verdict rule: supported if the predicted direction holds for both training seeds and the paired 95% bootstrap interval excludes 0 for both (for the original draft alone, its single interval); contradicted if the opposite holds under the same rule; otherwise inconclusive.
- P1. Both distilled drafts have lower forward KL and higher 1 - TV than the original on bank (i). Rival: no change, because 40,000 tokens of attention-only LoRA cannot move the next-token distribution measurably.
- P2. The on-policy minus off-policy difference in 1 - TV is larger on student-produced banks (bank (ii) and the own-student banks) than on bank (i). Rival: the difference is the same on every bank, because 0.5B and 1.5B Instruct prefixes are similar enough that prefix source does not matter.
- P3. The gain in 1 - TV over the original is smaller on shifted short-form templates than on familiar ones. Rival: equal gains, because what is learned is not template-specific.
- P4. In speculative trajectories, acceptance at proposal depth >= 2 (given earlier acceptance) exceeds depth-1 acceptance, so the i.i.d. alpha model with depth-1 alpha underpredicts the mean accepted prefix. Rival: acceptance is independent of depth.
- P5. Measured c >= 0.5 on the T4 at batch 1, and the original draft's measured S_4 < 1. Rival: c < 0.5 and S_4 > 1.
- P6. Measured S_gamma is below S_hat_gamma for every draft and gamma. Rival: |S - S_hat| / S_hat < 5%.
- P7. For every draft, S_2 >= S_4. Rival: S_4 > S_2 for the distilled drafts, because higher acceptance makes longer proposals pay.

## 6. Order of work
1. Sampler, toy test, negative control, cache, bonus, EOS and cap tests (CPU), committed.
2. Real-model preflight (Kaggle).
3. Banks (i), (ii), then (iii); task3/fixed64_ids.json and the scorer committed before any held-out evaluation.
4. Teacher collection, the four training runs, merging and merged-versus-unmerged checks.
5. Own-student banks, bank evaluation, speed benchmark, fixed-64 diagnostic, standalone accuracy.
6. Analysis from committed results only.

## AM1 (2026-11-28, before any real-model run)
- Preflight pass rule: for both models in FP16, max abs logit difference <= 0.25 and max TV <= 0.01 between split-prefill and full forward and between crop-and-reforward and fresh forward, on the first 2 GSM8K and first 2 short-form training prompts extended by up to 32 sampled tokens, with cache lengths checked after every speculative round.
- If a residual is all zero the sampler falls back to p and records it; this event has probability zero under exact arithmetic and is counted in every run.
- Learning rate: constant 3e-4 for all 157 updates (the manual specifies no schedule or warm-up).
- Held-out guard: a committed hash list of the 120 held-out prompts; training and collection processes refuse to load any listed item and record every held-out file open in their run JSON.
- Own-student banks (D14) are generated in the same notebook as training, right after each run's merge.
