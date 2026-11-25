# SPDX-License-Identifier: Apache-2.0
"""Integration test for NVIDIA Nemotron 3.5 Lightning 30B-A3B (NemotronH), served through vLLM on Neuron (Trn2).

Serves the real checkpoint on one Trainium2 device (TP=4, logical NeuronCore config 2) and checks a factual answer,
arithmetic, greedy determinism, a batch of different requests (each request's Mamba state stays its own), a long
prompt prefilled in several chunks (state carried across chunks), and a GSM8K subset.

Install both packages so that the TP worker processes see the model and the recurrent-state runner:

    pip install --no-deps --force-reinstall -e .
    pip install --no-deps --force-reinstall -e contrib/models/nemotron_h
    pip install datasets                       # for the GSM8K subset (downloads openai/gsm8k)
    NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_BACKEND=neuron_native \\
        pytest contrib/models/nemotron_h/test/integration/test_model.py -v -s --timeout=7200

On a trn2.48xlarge, also set NEURON_VISIBLE_DEVICES=0-3 to pick one device. Starting the engine compiles the
graphs (several minutes with a warm NEFF cache, longer on the first run).

Environment overrides: NEMOTRON_H_MODEL_PATH (checkpoint), NEMOTRON_H_GSM8K_N (questions, default 100).
"""

import glob
import os
import re

import pytest

MODEL_PATH = os.environ.get("NEMOTRON_H_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
GSM8K_N = int(os.environ.get("NEMOTRON_H_GSM8K_N", "100"))
# A small prefill chunk, so the long-prompt test crosses several chunks.
CHUNK = 2048

pytestmark = pytest.mark.skipif(not glob.glob("/dev/neuron*"), reason="needs Neuron devices (Trn2)")


@pytest.fixture(scope="module")
def llm():
    from vllm import LLM

    return LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=CHUNK,
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [CHUNK],
                "kv_segment_size_buckets": [CHUNK],
                "num_seqs_buckets": [1, 8],
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )


def _chat(llm, prompts, max_tokens=64):
    """Greedy chat completions with thinking off, one output text per prompt."""
    from vllm import SamplingParams

    outs = llm.chat(
        [[{"role": "user", "content": p}] for p in prompts],
        SamplingParams(temperature=0.0, max_tokens=max_tokens),
        chat_template_kwargs={"enable_thinking": False},
    )
    texts = [o.outputs[0].text.strip() for o in outs]
    for p, t in zip(prompts, texts):
        print(f"Q: {p[:80]!r} -> {t[:120]!r}")
    return texts


def test_capital(llm):
    assert "paris" in _chat(llm, ["What is the capital of France? Answer in one word."])[0].lower()


def test_arithmetic(llm):
    assert "56" in _chat(llm, ["What is 7 times 8? Answer with the number only."])[0]


def test_greedy_is_deterministic(llm):
    prompt = ["List three primary colors, comma separated."]
    assert _chat(llm, prompt) == _chat(llm, prompt)


def test_batch_of_different_requests(llm):
    # Eight different requests in one batch: a request that read another one's Mamba state would answer wrongly.
    pairs = [("France", "paris"), ("Japan", "tokyo"), ("Italy", "rome"), ("Germany", "berlin"),
             ("Spain", "madrid"), ("Canada", "ottawa"), ("Egypt", "cairo"), ("Kenya", "nairobi")]
    texts = _chat(llm, [f"What is the capital of {c}? Answer in one word." for c, _ in pairs], max_tokens=16)
    for (country, city), text in zip(pairs, texts):
        assert city in text.lower(), f"{country}: {text!r}"


def test_long_prompt_chunked_prefill(llm):
    # About 5,000 tokens: three prefill chunks of 2,048, with the fact near the start, so it must survive the
    # Mamba state and KV carried across chunks.
    filler = " ".join(f"item{i}" for i in range(2400))
    prompt = f"The secret word is BANANA. Here is a list: {filler}\nWhat is the secret word? Answer in one word."
    assert "banana" in _chat(llm, [prompt], max_tokens=16)[0].lower()


GSM8K_PROMPT = (
    "Solve the following math problem step by step. When you are done, write the final "
    "numeric answer on its own last line in exactly this form: Final answer: <number>\n\n"
    "Problem: {q}"
)
NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def _answer(text):
    m = re.findall(r"final answer\s*(?:is)?\s*[:\-]?\s*[\*\$\\\(\[\{ ]*(" + NUM + ")", text, flags=re.I)
    nums = m or re.findall(NUM, text)
    try:
        return float(nums[-1].replace(",", "").rstrip(".")) if nums else None
    except ValueError:
        return None


def test_gsm8k_subset(llm):
    datasets = pytest.importorskip("datasets")
    rows = datasets.load_dataset("openai/gsm8k", "main", split="test").select(range(GSM8K_N))
    texts = _chat(llm, [GSM8K_PROMPT.format(q=r["question"]) for r in rows], max_tokens=1024)
    gold = [float(r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]
    correct = sum(a is not None and abs(a - g) < 1e-6 for a, g in zip(map(_answer, texts), gold))
    acc = correct / GSM8K_N
    print(f"GSM8K first {GSM8K_N}: {acc:.3f}")
    # The full test set (1,319 questions) scores 94.5 % with this prompt on Trn2 (H100, stock vLLM: 95.0 %).
    assert acc >= 0.85, f"GSM8K accuracy {acc:.3f} on the first {GSM8K_N} questions"
