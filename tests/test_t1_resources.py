import pytest

from task1 import config as C
from task1.data import read_sequences
from task1.resources import EXECUTION, latency, prompt_ids


@pytest.fixture(scope="module")
def prompts():
    return prompt_ids(read_sequences("c4_calibration", limit=4))


def test_prompt_ids_fixed(prompts):
    seqs = read_sequences("c4_calibration", limit=4)
    assert set(prompts) == {512, 2048}
    assert prompts[512]["token_ids"] == seqs[0]["token_ids"]
    assert prompts[512]["source_rows"] == [seqs[0]["source_row"]]
    assert prompts[2048]["token_ids"] == [t for s in seqs for t in s["token_ids"]]
    assert prompts[2048]["source_rows"] == [s["source_row"] for s in seqs]


def test_latency_cpu(tiny_fp32, prompts):
    r = latency(tiny_fp32, prompts, warmup=1, reps=5, decode_steps=8)
    assert r["execution"] == EXECUTION and r["execution"]["mode"] == "dense_fp16"
    assert r["timer"] == "perf_counter" and r["batch_size"] == 1 and r["eval_id"] == C.EVAL_ID
    assert set(r["measurements"]) == {"prefill_512", "prefill_2048", "decode_8_after_prefill_512"}
    for m in r["measurements"].values():
        assert m["unit"] == "ms" and m["repetitions"] == 5 and len(m["times_ms"]) == 5
        assert all(t > 0 for t in m["times_ms"])
        assert min(m["times_ms"]) <= m["median_ms"] <= max(m["times_ms"])
        assert len(m["peak_allocated_bytes"]) == 5
    assert len(r["measurements"]["decode_8_after_prefill_512"]["decoded_token_ids"]) == 8
    with pytest.raises(AssertionError):
        latency(tiny_fp32, prompts, warmup=0, reps=4)
