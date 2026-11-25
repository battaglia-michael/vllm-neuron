# Contrib Model: NVIDIA Nemotron 3.5 Lightning 30B-A3B (NemotronH, text)

`NemotronHForCausalLM`, a hybrid of 23 Mamba-2 layers, 23 mixture-of-experts layers (128 experts, top-6, non-gated
squared-ReLU, plus a shared expert) and 6 attention layers without positional encoding. It runs on one Trainium2
device at TP=4 with BF16 weights, on the native PyTorch backend, with six model-specific NKI kernels in `src/kernels/`.

The Mamba layers need per-request recurrent state in the V1 model runner. This model builds on the recurrent-state
core from [PR #68](https://github.com/vllm-project/vllm-neuron/pull/68) (Qwen3.5), which must be merged first; see
[Changes outside this folder](#changes-outside-this-folder). The architecture, kernel choices and feature status are
in [`src/README.md`](src/README.md).

## Usage

Install the plugin from this branch and this folder as a vLLM plugin, into the DLAMI's vLLM environment:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
pip install --no-deps --force-reinstall -e .                          # vllm-neuron with the recurrent-state runner
pip install --no-deps --force-reinstall -e contrib/models/nemotron_h  # registers NemotronHForCausalLM
```

`--force-reinstall` is needed because the branch has the same version number as the DLAMI's plugin; `--no-deps` keeps
the DLAMI's pinned torch, vllm, transformers and nki. The package's `vllm.general_plugins` entry point registers the
architecture in every vLLM process (API server, engine, TP workers), replacing vLLM's GPU implementation.

### Serve with the OpenAI-compatible server

Up to 128 requests of up to 8,960 tokens on one device (on a trn2.48xlarge, `NEURON_VISIBLE_DEVICES` picks the
device's 4 NeuronCores):

```bash
export NEURON_VISIBLE_DEVICES=0-3 NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_BACKEND=neuron_native
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.25 NEURON_RT_EXEC_TIMEOUT=300
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --served-model-name nemotron \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 8960 --max-num-seqs 128 --max-num-batched-tokens 4096 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [2048, 4096],
     "kv_segment_size_buckets": [4096], "num_seqs_buckets": [1, 8, 32, 128]}}'
```

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nemotron", "max_tokens": 32, "temperature": 0,
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
  "chat_template_kwargs": {"enable_thinking": false}}'
```

For up to 32 requests of up to 40,960 tokens use `--max-model-len 40960 --max-num-seqs 32`,
`"num_seqs_buckets": [1, 8, 32]` and `"decode_context_length_buckets": [2048, 10240]` (KV cap 0.25). For tool
calling add `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`.

| Setting | Why |
|---|---|
| `NEURON_VISIBLE_DEVICES` | Selects cores by id. `NEURON_RT_VISIBLE_CORES` fails with `cannot be used with multi-processing execution on vLLM`. |
| `NEURON_SKIP_EFA_AFFINITY=1` | Without EFA interfaces the engine fails at start with `No EFA device found`. |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.25` | The scheduler admits at most `kv_cache_tokens // max_model_len` requests; 0.25 holds 128 x 8,960 tokens and still leaves HBM for the graphs. |
| `NEURON_RT_EXEC_TIMEOUT=300` | The default 30 s timeout killed the engine (`NRT_TIMEOUT`) while the host was compiling other servers. |
| `kv_segment_size_buckets` | Chunked-prefill attention uses `NF.segmented_attention`; without it a torch fallback builds large fp32 score tensors. |
| No `decode_context_length_buckets` at 128 requests | With a 2,048 context bucket the graphs did not fit in HBM at load (`Allocation Failure`). |

vLLM's torch.compile cache is keyed by the vLLM config only, not by the model source or the `NEMOTRON_H_*`
environment. After changing either, start with a fresh `VLLM_CACHE_ROOT`.

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=8192,
    max_num_seqs=8,
    max_num_batched_tokens=2048,
    enable_prefix_caching=False,
    additional_config={"neuron_config": {
        "num_batched_tokens_buckets": [2048], "kv_segment_size_buckets": [2048], "num_seqs_buckets": [1, 8]}},
)
out = llm.chat([{"role": "user", "content": "What is the capital of France?"}],
               SamplingParams(temperature=0.0, max_tokens=32),
               chat_template_kwargs={"enable_thinking": False})
print(out[0].outputs[0].text)
```

The model is registered by the installed package's entry point, so no `ModelRegistry` call is needed, also with
`tensor_parallel_size` > 1.

## Compatibility Matrix

| Instance | Neuron SDK | vllm-neuron | vLLM | Configuration | Status |
|---|---|---|---|---|---|
| trn2.48xlarge | 2.33 (DLAMI `Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`) | 0.29.0.1.2.0 + PR #68 + this folder | 0.29.0 | one device per server, TP=4, LNC=2, BF16 | Tested (November 2026); up to 15 servers measured together on one instance |
| trn2.3xlarge | 2.33 | same | 0.29.0 | TP=4, LNC=2 | Not tested (one device; should fit: 63 GB of weights in 96 GB HBM) |
| Trn3 | | | | | Not tested |

## Accuracy

One Trainium2 device with the serve configuration above (greedy, thinking off), against Hugging Face transformers and
against stock vLLM 0.27.1 on one H100 with NVIDIA's recipe:

| Test | Trainium2 | Reference |
|---|---|---|
| Teacher-forced greedy token match vs HF FP32, 900 positions up to 8K tokens | 98.44 % | HF's own BF16 model: 98.4 %; H100: 98.34 % |
| Batch invariance (each prompt alone vs in a batch of 32) | pass | |
| Chunked prefill vs HF labels | pass (chunk 0: 98.6 %, chunk 1: 99.0 %) | |
| GSM8K, 1,319 questions, zero-shot CoT | 94.47 % | H100: 95.00 % |
| Tool calling, BFCL simple + multiple, 200 items | 83.0 % | H100: 84.0 % |
| Needle retrieval at 1K / 16K / 32K / 128K | 20 / 20 | H100: 20 / 20 |

The Mamba step size is not clamped (`NEMOTRON_H_DT_CLAMP=none`, the default), as in NVIDIA's model code and vLLM on
GPU. Hugging Face transformers clamps it in its chunked scan but not in cached decode, so HF reference labels must
be made with the clamp removed (see `src/README.md`, "dt clamp").

Performance on one device (1,024 in / 1,024 out): 154 output tok/s at one request (6.3 ms per token), 789 tok/s at
32. The full benchmark and the H100 comparison are in
https://github.com/battaglia-michael/neuron-things/tree/main/nemotron-3.5-lightning-vllm-neuron.

## Example Checkpoints

- [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) (tested)

## Testing

### Integration test (Trn2)

Serves the checkpoint on one device and checks a factual answer, arithmetic, greedy determinism, a batch of eight
different requests, a 5,000-token prompt prefilled in three chunks, and the first 100 GSM8K questions (>= 85 %):

```bash
pip install datasets
NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_BACKEND=neuron_native \
  pytest contrib/models/nemotron_h/test/integration/test_model.py -v -s --timeout=7200
```

### Unit tests (CPU)

Build a reduced checkpoint first (real weights, layers 0, 1, 5, 7 = Mamba, MoE, attention, Mamba):

```bash
python contrib/models/nemotron_h/scripts/make_reduced_checkpoint.py $MODEL_DIR /tmp/nemotron-h-4l-fp32 --layers 0,1,5,7 --dtype float32
NEMOTRON_H_TEST_CKPT=/tmp/nemotron-h-4l-fp32 VLLM_NEURON_CPU_MODE=1 pytest contrib/models/nemotron_h/test/unit -v
```

These compare each layer type with Hugging Face (prefill, padded prefill and decode, chunked prefill on both paths,
batched decode). Two more CPU checks:

- TP=4 simulated with 4 gloo processes: `VLLM_NEURON_CPU_MODE=1 python contrib/models/nemotron_h/test/unit/tp_sim.py --ckpt /tmp/nemotron-h-4l-fp32 --tp 4`.
- The whole model through vLLM in CPU mode (runner, KV cache, state pool, chunked prefill, batching):
  `contrib/models/nemotron_h/scripts/run_cpu_numerics.sh <ckpt_fp32> <ckpt_bf16> <out> <logs>`. CPU mode needs
  `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.002`, or it allocates a KV cache sized for the whole host memory.

Measured on the 4-layer checkpoint: vLLM CPU mode matches HF FP32 to 5.5e-5 with every greedy token identical, alone
and batched; TP=4 simulation to 1.8e-5. The `test/unit/device_*.py` files are kernel-level checks that need a Neuron
device.

## Changes outside this folder

The model needs these framework changes, in `vllm_neuron/` (in addition to PR #68's recurrent-state core):

| File | Change |
|---|---|
| `vllm/worker/neuron_model_runner.py` | Host-side cache of per-step decode inputs (`VLLM_NEURON_HOST_PREP_CACHE`, default on: block-table clones, state-pool table and sampling-parameter tensor are reused while unchanged); recurrent state bound into the shared KV pool for prefix caching (`VLLM_NEURON_STATE_KV_ALIAS`); optional power-of-2 state-pool pages (`VLLM_NEURON_STATE_POOL_POW2_PAGES`, default off). |
| `vllm/worker/neuron_worker.py` | With the state in the shared pool, clears the pool after the prefill warmup (`VLLM_NEURON_WARMUP_CLEAR_SHARED`, default `auto`), so decode warmup padding rows don't read warmup K/V as Mamba state. |
| `envs.py` | Registers `VLLM_NEURON_HOST_PREP_CACHE`. |
| `functional/attention/attention_decode.py` | `VLLM_NEURON_ATTN_DECODE_TORCH=1` forces the torch decode fallback (diagnostic, default off). |

The host-prep cache helps any model on the V1 runner (it removed host-side delays between decode steps here); the
prefix-caching changes apply to any model with recurrent state. They can be split into their own PR if preferred.
