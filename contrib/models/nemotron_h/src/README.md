# NemotronH: architecture and feature status

`NemotronHForCausalLM` (hybrid Mamba-2 + MoE + attention) for the vLLM Neuron
plugin's V1 runner, built on the recurrent-state core (`StateLayerSpec`,
`KVSpec.state_layers`, `HasRecurrentState`, `bind_state_cache`, dedicated state
pool) that PR #68 added for Qwen3.5. Target checkpoint:
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

Implementation card for `contrib/models/nemotron_h`. The folder's `README.md` has usage, the compatibility
matrix, accuracy and tests. IDs such as `T12` or `R2-03` below and in code comments name optimization steps;
the write-up's "How it got faster" table summarizes them
(https://github.com/battaglia-michael/neuron-things/tree/main/nemotron-3.5-lightning-vllm-neuron).

## Architecture (30B-A3B `config.json`)

| Part | Value | Per rank at TP=4 |
|---|---|---|
| Layers | 52: 23 Mamba-2, 23 MoE, 6 attention (`layers_block_type`); 1 MTP layer (not loaded) | |
| Block | `x + mixer(RMSNorm(x))`, one mixer per layer, no separate MLP | |
| hidden / vocab | 2688 / 131072, untied `lm_head` | vocab 32768 |
| Attention | 32 Q / 2 KV heads, head_dim 128, no bias, **no positional encoding** | 8 Q heads, 1 KV head (each KV head on 2 ranks) |
| Mamba-2 | 64 heads x 64, d_state 128, n_groups 8, conv kernel 4 (bias), chunk 128, gated RMSNorm over groups of 512 | 16 heads, 2 groups; state conv `(3, 1536)` + ssm `(16, 64, 128)` fp32 |
| MoE | 128 experts, top-6, sigmoid router + `e_score_correction_bias` (selection only), `norm_topk_prob`, `routed_scaling_factor` 2.5, n_group 1; experts **non-gated** `relu(x)^2` (`up_proj`, `down_proj`), intermediate 1856; 1 shared expert, intermediate 3712 | experts 464, shared 928 |
| Norm eps | 1e-5 (`layer_norm_epsilon`) | |

## Differences from the reference ports

Structure follows `contrib/models/qwen3_5` (state plumbing, prefill/decode split,
sequence parallelism, weight-loader checks) and `vllm_neuron/model/llama3`
(attention on `NF.*`). Model-specific points, all taken from HF
`modeling_nemotron_h.py`:

* **No RoPE.** `NemotronHAttention` never rotates q/k (vLLM's GPU model doesn't
  either); `rope_theta` in `config.json` is unused. `NF.qkv_proj` and
  `NF.attention_decode` are called with no cos/sin.
* **RMSNorm** is `weight * normalize(x.float()).to(dtype)` (weight applied in the
  model dtype), not the zero-centred Qwen3.5 form.
* **Gated norm** gates before normalising (`norm_before_gate=False`):
  `y * silu(z)`, then RMS over groups of `intermediate / n_groups` = 512 channels.
  A group's 8 heads live on one rank, so the norm needs no collective.
* **dt clamp (decided in T03b: no clamp, `NEMOTRON_H_DT_CLAMP=none` is the
  default).** Stock `transformers` 5.17 `NemotronHMamba2Mixer` hard-codes
  `time_step_limit = (time_step_min, inf)` and clamps `softplus(dt + dt_bias)` to
  1e-3 in its chunked scan (every full-sequence forward) but not in its cached
  single-step decode. Everything else leaves dt unclamped: the checkpoint's
  `config.json` (no `time_step_limit`, so the config default `(0.0, inf)`),
  NVIDIA's own remote `modeling_nemotron_h.py` for earlier Nemotron-H models
  (`self.time_step_limit = config.time_step_limit`), Megatron-LM's `MambaMixer`
  (training code: `time_step_min` only sets the `dt_bias` init range; the scan
  passes `dt_limit=(0.0, inf)` or nothing, decode applies only softplus),
  `mamba_ssm`'s `Mamba2` default, and vLLM GPU (`mamba_mixer2.py`
  `dt_limit=(0.0, inf)` in v0.27.1 and v0.29.0), which is the H100 baseline.
  The weights agree: the `dt_bias` init (Megatron and HF) puts every head's
  `softplus(dt_bias)` in [1e-3, 0.1], and after training 133 of the 1,472 Mamba
  heads sit below 1e-3, down to 1.1e-6 (`dt_bias` -13.7). A 1e-3 clamp during
  training would have zeroed the gradient that moved them there.
  Measured on the full model (T03b):
  HF fp32 with and without the clamp pick different argmax tokens at about 5% of
  teacher-forced positions, more at long context. Without the clamp HF predicts
  the real next token more often (McNemar z = 4.9). GSM8K (greedy, thinking off) on
  trn2 is 94.84% (1251/1319) with either setting; the first 200 give 96.0% without
  and 95.5% with the clamp. The device port matches
  whichever HF variant uses its own setting. `NEMOTRON_H_DT_CLAMP=hf` is kept as
  an option: it clamps in prefill and decode, so decode equals stock HF's
  full-sequence forward. `nemotron_h.mamba.align_hf_reference(model)` patches an
  HF model to the port's setting. Every test that compares against HF calls it
  (`test/unit/*`, `test/integration/cpu_numerics.py`, which `run_cpu_numerics.sh` runs).
  With `none` on the 4-layer model: fp32 vs
  aligned HF has a max |logit diff| of 5.5e-5 with every greedy token matching,
  and 8 decode steps give 3.3e-5. Against stock HF the same 8 decode steps
  differ by 6.9.
* **D skip** uses the post-conv `x` (before multiplying by dt), as HF.
* **MoE router** is hand-rolled (see the kernel table). Top-k uses
  `vllm_neuron.functional.topk.topk`.

## Kernel selection

| Block | Call | Notes / blocker |
|---|---|---|
| RMSNorm | torch (`NemotronHRMSNorm`) | HF form; not fused into qkv/router yet |
| QKV projection | `NF.qkv_proj` | no RoPE caches passed |
| Prefill attention | `NF.flash_attention` | |
| Chunked-prefill attention | `NF.qkv_proj(use_block_kv=True)` + `NF.segmented_attention` when `kv_segment_size_buckets` is set; otherwise a torch prefix+chunk attention (from the Qwen3.5 contrib) | segment sizes supported by the runner: 512, 1024, ..., 8192 |
| Decode attention | `NF.attention_decode` (QKV + attention + o_proj, in-kernel KV write) | `cos=None` (no RoPE)  `NEMOTRON_H_DECODE_CTX_ALIGN=1024` (default, R2-02): the block table handed to the kernel is padded with masked `-1` columns to a multiple of 1,024 tokens. nkilib `attention_tkg` shrinks its block length to gcd(32, context / 128, ...), which is 1-2 tokens (256-512 B DMA elements) at the S profile's 8,960; at 9,216 it is 4-8. S decode step b32 40.0 -> 35.4 ms, b128 82.8 -> 63.9 ms. `0` = off |
| O projection (prefill) | `NF.o_proj` | |
| Mamba in_proj / out_proj | torch matmul | Prefill (T12): `NEMOTRON_H_MAMBA_PREFILL=split` computes `z`, `xBC` and `dt` with three matmuls on column slices of the stored `in_proj` weight instead of one matmul followed by column slices of its `[T, 2576]` output. With one matmul, neuronx-cc re-lays the output out with ~20k 8-element bf16 copies and 32x32 transposes on one physical core (5.5 of 15.6 ms of a 2,048-token layer in the device profile). One layer 16.9 -> 2.8 ms; full-model prefill chunk 897 -> 330 ms (Mamba 622 -> 57 ms over 23 layers); the slower later blocks of T08 (27.6 vs 14.8 ms) go away with it. `base` = the single matmul (baseline-v1..opt2). Decode keeps the single matmul (split: batch 1 0.66 -> 1.06 ms per layer, batch 32 3.77 -> 2.68, batch 128 9.16 -> 3.51; not wired). |
| Mamba conv1d | torch, unrolled taps (Qwen3.5 pattern) | NKI Library `conv/depthwise_conv1d` is a later optimization |
| Mamba prefill scan | torch chunked SSD (chunk 128, rank-3 matmuls, Python loop over chunks for the state carry) | NKI Library `scan/ssd.py` requires `ngroups=1`; at TP=4 there are 2 groups per rank, so it would be called once per group (later) |
| Mamba prefill, fused kernel (R2-03) | `kernels/ssd_prefill.py` (A06 kernel 2 v2, split-input variant) on the whole prefill chunk: causal conv from the cached window + bias + SiLU, `softplus(dt + dt_bias)` masked to real tokens, chunked SSD scan (chunk 128, exponents `min(acum_i - acum_j, 0)`, fp32 state kept transposed), `D x` and the gated RMSNorm, one SSM group per physical core; torch keeps the three split in_proj matmuls, `out_proj` and the conv-window tail. The conv weight is passed as `[K, conv_dim]` (`NEMOTRON_H_MAMBA_PREFILL_CONV_KMAJOR=1`): reading `[conv_dim, K]` with a stride-K pattern made ~13k 600-B DMAs per layer in the compiler's repeated module | `NEMOTRON_H_MAMBA_PREFILL_KERNEL` (default 1 since R2-03; `0` = the torch path on 1,024-token sub-chunks). Also used with the KV-aliased shared state pool (prefix caching) since R2-03p2 (`NEMOTRON_H_MAMBA_PREFILL_KERNEL_SHARED`, default 1; `0` = torch there): the P hang in the batch-8 decode warm-up was triggered by the warm-up prefill's out-of-range (dropped) null-page state write, so that write now goes to page 0 (`NEMOTRON_H_PREFILL_NULL_WRITE=page0`, default; `oor` = old) and the worker clears the shared pool before the decode warm-up (`VLLM_NEURON_WARMUP_CLEAR_SHARED=auto`); prefix_8k c32 (development devices) TTFT -14..-18%, tok/s +1.5..+5.2%. Full-model 4,096-token chunk 372.9 -> 297.9 ms (Mamba 125.1 -> 57.6 ms); kernel vs torch on device 99.98% bit-identical, state closer to float64; gate passes; device-0 TTFT -17..-20% on every Must row (optimization step R2-C) |
| Mamba prefill in_proj kernel (R2-03m) | `kernels/ssd_prefill.py mamba_in_proj_kernel`: the kernel path's `in_proj` (`[T, 2688] @ [2688, 2576]` per rank) as one NKI matmul kernel (LNC=2 column split, weight slice resident in SBUF, TensorE transposes of the token tiles, fp32 PSUM accumulation), then the proj-input fused kernel (`mamba_prefill_fused_kernel`, conv weight `[K, conv_dim]`, no xBC mask). In the compiler's repeated module the torch in_proj matmuls were lowered with strided moving operands and ran under the TensorE activity throttle (k/n 4/8): 1.5-1.8 ms per layer vs 0.55 ms in module 0 | `NEMOTRON_H_MAMBA_PREFILL_INPROJ` (default `nki` since R2-03m; `torch` = R2-03's three split matmuls). Opt-in, not default: `NEMOTRON_H_MAMBA_PREFILL_KSPLIT=2` (pieces; no overlap on device), `NEMOTRON_H_MAMBA_PREFILL_SCAN=sp1,zearly,convpool` (scan-phase ordering / engine options; convpool is slower). Aux-device profile, 4,096-token chunk: Mamba 58.0 -> 38.2 ms, chunk 284.6 -> 261.9 ms; numbers and gate in the development notes |
| Batch-1 MoE router (R2-10) | `kernels/router_top8.router_fused`: for one decode token, `x` -> fp32 router matmul on the Tensor Engine -> sigmoid -> + bias -> `max8` + `nc_find_index8` (top 6) -> normalized, scaled weights and expert row offsets, all in SBUF; the two outputs go straight into `kernels/moe_topk.py`, which now also issues every expert-tile load before its `w` / `x` broadcast loads (those on the Scalar hardware-DGE queue) | `NEMOTRON_H_ROUTER_FUSED`, `NEMOTRON_H_MOE_TOPK_EARLY`, `NEMOTRON_H_MOE_TOPK_BQ=act` (default since R2-10; `0` / `0` / `default` restore the torch router, the rotational top-k and the old load order; the FP8 kernel keeps the old order). The torch path hands the partition-major `[1, 128]` router vector to HBM as ~126 four-byte DMA descriptors (~9 us) and adds more hand-offs before the first expert load. Full-model batch-1 step 6.75 -> 6.34 ms (device profile), first expert DMA 45 -> 30 us into the MoE segment; gate passes; device-0 c=1 Must rows faster in ABBA pairs (optimization step R2-D) |
| Mamba decode step | fused NKI kernel `kernels/mamba_decode.py` (`NEMOTRON_H_MAMBA_DECODE_KERNEL=1`, default; `0` = torch) | The kernel does everything between `in_proj` and `out_proj` for up to 128 requests: state page read, conv window shift + depthwise conv + bias + SiLU, `softplus(dt + dt_bias)` (no clamp; the kernel path is used only with `NEMOTRON_H_DT_CLAMP=none`), `exp(dt*A)`, the fp32 SSM update written back in place, `y = C.h + D x`, and the gated RMSNorm. One SSM group per physical core (LNC=2 grid). x/B/C are rounded to bf16 as in the torch path; the rest is fp32, so it is closer to an fp32 reference than the bf16 torch path (A05: device output max abs vs fp32 <= 9.8e-4 vs 7.8e-3 for torch). One Mamba layer on device, flag off -> on: 0.96 -> 0.12 ms at batch 1, 3.6 -> 0.58 ms at batch 32, 9.7 -> 1.65 ms at batch 128 (A05 unit test, including in_proj/out_proj). On by default since T15 (`NEMOTRON_H_MAMBA_DECODE_KERNEL=0` restores the torch path). Main lane, one trn2 device, on top of opt3: batch-32 decode step 123.5 -> 42.4 ms (Mamba 88.5 -> 9.2 ms over 23 layers), batch 1 23.4 -> 19.5 ms; c32 output tok/s +64% (8K/512) to +209% (1K/8K); correctness gate passes (teacher-forced, chunked prefill, batch invariance), GSM8K 94.69% |
| Mamba decode kernel state DMAs | `NEMOTRON_H_MAMBA_DECODE_DMA=v3` (default) or `v2` | Each physical core moves its SSM group's state (256 KiB per request) with dynamic-offset DMAs. The second core's group straddles state halves A and B, which takes 2 DMAs per request. `v2` puts every state DMA on the Sync hardware-DGE queue. `v3` moves the second core's half-A piece to software DGE. Outputs are bit-identical. One layer's `forward_decode` on device, v2 -> v3: 0.445 -> 0.375 ms at batch 32, 1.39 -> 1.11 ms at batch 128, unchanged at batch 1 (A05b, chained-graph timing). Aux device, profile S: decode ITL 42.5 -> 40.8 ms at c32 and 92.8 -> 86.2 ms at c128; gate passes. Per-partition vector-offset DMAs and splitting each core between both DGE paths were slower: both paths end on the core's one DMA engine |
| Mamba state page layout (T-mamba-state-align) | `NEMOTRON_H_STATE_ALIGN=1` (default; `0` = previous layout). At TP=4 the old layout `[conv 36 rows \| SSM 1024 rows]` put SSM group 1 across state halves A and B (92 + 420 rows), so the decode kernel's second core needed 2 state DMAs per request and direction (23 + 105 partitions). The aligned page is A = `[conv 36 \| group 0 512 \| pad 92]`, B = `[group 1 512 \| pad 128]` (same page size), so each core moves its group with one 128-partition DMA. Prefill reads the aligned page with one gather kernel (`state_cache.gather_state_rows`; a torch concatenate of two slices made prefill Mamba 3.6x slower). Falls back to the old layout with a warning when the aligned one does not fit (TP=1) | Unit (same process): kernel 0.290 -> 0.258 ms (batch 32), 1.072 -> 0.917 ms (batch 128), outputs bit-identical. Full-model live trace: batch-32 step 40.83 -> 40.01 ms, batch 128 86.18 -> 82.80 ms, batch 1 unchanged (7.04 / 7.02 ms); greedy outputs identical to the old layout; gate passes; device 0 ITL p50 -1.9% (c32), -3.9% (c128) over 2 same-session pairs (optimization step 14). With prefix caching (KV-aliased shared pool, T11c/T11d) the layout applies too since T11e: the shared page has 1,024 rows per half at TP=4 with 2,048-token blocks, so A = `[conv 36 \| group 0 512 \| pad 476]`, B = `[group 1 512 \| pad 512]`; `NEMOTRON_H_STATE_ALIGN_SHARED=0` keeps the old layout in the shared pool only. prefix_cache_gate passes with outputs identical to the old layout; prefix_8k c32 on device 0, 2 same-session pairs: 627.3 / 628.7 -> 648.9 / 648.4 tok/s, TPOT p50 43.4 -> 41.9 ms; default-path (prefix caching off) graphs identical |
| State page read/write | `kernels/state_cache.py` (copied from PR #68) on device, `index_select`/`index_copy_` on CPU | `NEMOTRON_H_STATE_KERNEL=1` (default, read+write), `read`, `write`, `0` |
| MoE router (prefill, decode buckets > 1) | torch (sigmoid, + bias for selection, top-k, renorm, scale) | **HAND-ROLLED**: `NF.router`'s `router_bias` is added to the logits before the activation and the activation output is used for both selection and weights; there is no selection-only bias. `RouterComputationOrder` has only `PRENORM_LINEAR_{TOPK_ACT_SCATTER, ACT_TOPK_RENORM_SCATTER, TOPK_SCATTER_ACT}`. |
| MoE top-k | `vllm_neuron.functional.topk.topk` | NKI where it fits, else `torch.topk` (logged `kernel=torch ... reason=no-nki-method` on CPU) |
| MoE experts | torch: `relu(F.linear(x, W_up_all))^2 * affinity @ W_down_all` over all 128 experts (two dense matmuls). `W_up_all` is stored `[E*I, H]` (checkpoint `[out, in]` orientation), `W_down_all` `[E*I, H]` (`[in, out]`) | **torch-fallback**: `NF.moe_block_tkg` accepts only SiLU/GELU/Swish gated experts, and `NF.moe_cte` does not expose nkilib's `skip_gate_proj` (see below). Computes 128/6 = 21x the routed FLOPs, and every decode step reads all expert weights (~14.7 GB per rank at TP=4, an estimated ~20 ms/step floor at batch 1, about 9x a top-6 kernel's traffic; the gap shrinks with batch). `NEMOTRON_H_MOE_WT` (default `up`; `none` = baseline-v1 `[H, E*I]` up weight) picks the stored layout: with `none` the compiler inserts a transpose of the 319 MB up weight into every decode step (T08/T09), and decode at batch 1 takes 71.6 ms/token instead of 32.3. |
| MoE experts, decode batch 1 (T10) | top-6 only: `kernels/expert_gather.py` copies the 6 selected experts' up and down matrices out of the stacked `[E, I, H]` weights (one DMA each, runtime expert id as a scalar offset), then `F.linear` / matmul in torch on `[6*I, H]` | `NEMOTRON_H_MOE_TOPK_MAX_T` (default 1 with `MOE_WT=up`; 0 = dense everywhere). Full model, batch-1 decode 32.3 -> 23.4 ms/token (MoE 1.02 -> 0.68 ms per layer). The copies run on one DMA engine per core (~27 GB/s), so ~0.55 ms of each layer is waiting for them; at batch 8 (48 copies) it is slower than dense, so bucket 8+ and prefill stay dense. `NEMOTRON_H_EXPERT_GATHER_MODE` (hw/sw/mix) and `_SPLIT` made no measurable difference (all land on the same engine). |
| MoE experts, decode batch 1, fused (T14) | top-6 only, one NKI kernel `kernels/moe_topk.py`: for each (token, expert) pair the expert's up and down matrices stream from the stacked weights straight into SBUF (`[116, 2688]` row tiles, runtime expert offset, hardware DGE on the Sync and Scalar queues, spread over the DMA engines); `u = up_e x` as Vector Engine dot products, `relu(u)^2 * w`, down projection on the Tensor Engine accumulated in PSUM; the 6 pairs are split over the 2 physical cores, fp32 partials summed in torch | `NEMOTRON_H_MOE_TOPK_FUSED` (default 1; 0 = the T10 gather path). Full model, batch-1 decode 19.5 -> 9.4 ms/token (MoE 0.68 -> 0.23 ms per layer); c1 TPOT -51..-53% on every Must row, batch 8+ and prefill unchanged (dense). At batch 8 the per-pair loop is slower than dense (unit test 1.32 vs 1.18 ms per layer), so it stays batch 1 only. Gate passes. |
| MoE experts, decode batch 1, NKI Library (T09b) | top-6 via `nkilib.core.moe.moe_tkg` (selective-expert mode) with `skip_gate_proj` + `SquaredReLU`, `kernels/moe_tkg_sq.py`. nkilib 0.7 needs three patches, applied when the module is imported: `moe_tkg()` does not pass `skip_gate_proj`, the skip-gate activation rejects `SquaredReLU`, and the expert loops index `GateUpDim.UP` = 1 of the weight. Needs the routed up weights a second time as `[E, H, 1, I]` (in, out), `experts_up_tkg`: +6.84 GiB per rank (weights 14.74 -> 21.58 GiB) | `NEMOTRON_H_MOE_TKG` (default 0). Fits only in a latency profile with one request at a time (`bench/bringup/profiles/C1.sh`: max-num-seqs 1, decode bucket [1], KV cap 0.01). There, c1 TPOT is 9.0-9.4 -> 6.6-7.1 ms (the C1 profile with the T14 kernel is the control), gate passes with the same first-token outcome as T14 at all 894 windows. The S/L profiles cannot hold the copy. With the up weight stored `[E, H, I]` only, the dense path is 2.2x slower at batch 32 and 17% slower at T=2048 (unit, bench/opt/T09b) |
| MoE experts, decode batch 1, expert-tile DMAs (T10b) | same kernel as T14 (`kernels/moe_topk.py`), but the `[116, 2688]` expert row tiles are loaded with software DGE (GpSimd writes the descriptors, same scalar expert offset) instead of hardware DGE on the Sync/Scalar queues. A 4-layer device profile shows the hardware-DGE tiles on 4 of the 16 DMA engines of each physical core (~24 GB/s each); with software DGE they spread over all 16 (8 SBUF partitions per engine). Outputs are bit-identical | `NEMOTRON_H_MOE_TOPK_DMA` (`sw` default since T10b, `hw` = T14). Unit, 8 MoE layers at batch 1: 0.307 -> 0.199 ms/layer. 4-layer device profile: MoE block 0.253 -> 0.158 ms. Full model, batch-1 decode step 9.38 -> 7.37 ms (MoE 0.234 -> 0.148 ms per layer), c1 TPOT on device 0 9.1-9.4 -> 7.1-7.4 ms in the S/L profiles (no second weight copy). Gate passes with the same outcome as T14 at all 894 windows. Comparing kernel variants needs `NKI_ENABLE_TRACE_CACHE=0`: the persisted NKI trace cache ignores module globals |
| MoE experts, decode batch 1, expert-tile height (T14c) | same kernel, `[128, 2688]` expert row tiles (three per matrix plus one `[80, 2688]`, since I = 464 = 3 x 128 + 80) instead of `[116, 2688]`. DMA work is spread over the 16 engines of a physical core by partition, and 116 = 4 x 29 rows used 4 engines on hardware DGE and split unevenly on software DGE. Same `[E * I, H]` weights (no second copy, unlike NKI Library `moe_tkg`); the PSUM accumulation order of the down projection changes, so outputs are close but not bit-identical | `NEMOTRON_H_MOE_TOPK_TILE` (`128` default since T14c, `exact` = 116-row tiles). Unit device profile, 4 MoE layers: 131.3 -> 118.3 us per layer (NKI Library `moe_tkg` 140.4, T14 hardware DGE 220.7). Full model, batch-1 decode step 7.34 -> 7.04 ms (MoE 0.148 -> 0.134 ms per layer); device 0, c=1 TPOT -2.2% to -3.9% on the 4 Must rows over 2 server pairs, TTFT unchanged; gate passes |
| MoE experts, decode buckets 8 and 16, routed (R2-01) | `kernels/moe_decode_routed.py`: torch compacts the experts any token of the step picked (cumsum over the `[E]` mask + one 128-element scatter, static shapes); the NKI kernel walks that list in a dynamic loop, expert `e` on physical core `e % 2` in ascending id order, two entries per iteration with double-buffered software-DGE loads of the `[128/80, 2688]` up/down row tiles at a runtime row offset; per expert: up-weight transposes on the Tensor Engine, `relu(x up^T)^2 * w[:, e]` for all T tokens (w = 0 for tokens that did not pick e), down projection, float32 accumulation; torch adds the two cores' partials. The parity split makes a token's result independent of the other tokens' experts (unpicked experts add exact zeros), which the batch-invariance gate needs; a balanced position split failed it | `NEMOTRON_H_MOE_ROUTED_BUCKETS` (comma list of decode token counts; R2-01 measured `8,16`). Distinct experts per step on random-dataset prompts: 31 / 48 / 67 / 103 at batch 8 / 16 / 32 / 128. Full-model traces: batch-8 step 32.4 -> 20.9 ms (MoE 1.045 -> 0.567 ms/layer), batch 16 33.6 -> 25.8 ms (1.028 -> 0.663; late in decode 29.2 ms); batch 32 is slower late in decode (1.040 -> 1.163 ms/layer) and batch 128 equal, so they stay dense. ~25 us per expert per physical core against ~16 us for the dense path. R2-01b (`NEMOTRON_H_MOE_ROUTED_V2`, default `skip,bulk,p2,pre8:16,pre16:24,pre32:40`, `none` = the R2-01 loop): the dynamic loop drained the DMA queue at every 2-entry iteration (8-10 us idle each); the first K entries now run as straight-line code, pad loads are skipped (`oob_mode=skip`) and the static list is read in one DMA; bit-identical outputs; late-decode MoE per layer b8 0.571 -> 0.534 ms, b16 0.871 -> 0.744 ms; b32 1.066 vs dense 1.042, still dense. Gate passes (optimization step R2-A). `NEMOTRON_H_MOE_ROUTED_SPLIT=pos` (balanced split, not batch-invariant), `_PIPE=v2` (wrong results, do not use), `_COPY` (PSUM copy engines) are debug knobs |
| MoE experts, prefill (A07 kernel, T12b) | top-6 only, `kernels/moe_prefill.py`: static-shape torch dispatch of the (token, expert) pairs into 128-slot expert tiles; the NKI kernel splits the experts over the 2 physical cores, transposes each expert's up weight on the Tensor Engine, gathers each tile's token rows, runs up / relu² / down / routing weight and scatters the rows back; torch sums the k rows | `NEMOTRON_H_MOE_PREFILL_KERNEL` (default 1 with `NEMOTRON_H_MOE_WT=up`), `NEMOTRON_H_MOE_PREFILL_PAIR=sorted` (default; experts paired by tile count, runtime-offset weight loads, outputs bit-identical to `static`), `NEMOTRON_H_MOE_PREFILL_ACC=1` (default with `sorted`: k-sum in the kernel by DMA scatter-add into per-core fp32 planes; `0` = torch k-sum, bit-identical to `static`). One trn2 device, 2,048-token chunk: dense 330 ms → static 267.5 → sorted 258.3 → sorted+acc 248.8 ms; TTFT -15..-19% on every Must row (optimization step 8) |
| MoE experts, prefill, dispatch and first tile (T12d) | same kernel. The dispatch scatters only the slot -> pair map; the token of each slot is `pair // k` (exact float arithmetic) and the kernel adds its core's plane offset (two of three `[NPAD]` int32 scatters gone, ~0.6 ms per layer of serialized GpSimd DMAs). The first tile of every expert runs as straight-line code and only the rest in the dynamic loop, because each loop iteration ends with a barrier over all engines that waits for the tile's read-modify-write; every expert group gets at least one tile (empty groups an all-pad tile, skipped by the out-of-bounds RMW) | `NEMOTRON_H_MOE_PREFILL_SCATTER1` and `NEMOTRON_H_MOE_PREFILL_TILE0` (both default 1 since T12d, with ACC=1; `NEMOTRON_H_MOE_PREFILL_TILE0_MIN_T`, default 1024: smaller chunks keep the loop only). Unit (T=2048, random weights, 4 layers): 6.52 -> 6.28 -> 5.05 ms per layer, outputs bit-identical (also with adversarial routing: three consecutive same-core tiles on the same rows). Full model, 2,048-token chunk 248.8 -> 219.3 ms (MoE 168.8 -> 139.8); gate passes; device 0 TTFT -6..-12% (optimization step 12) |
| MoE experts, prefill, expert-tile height (T14d) | same kernel, `[128, 2688]` expert weight row tiles (three per matrix plus one `[80, 2688]`) instead of `[116, 2688]`, as T14c did for the batch-1 kernel. With 116 rows the software-DGE weight loads left 2 of the 16 DMA engines of a physical core mostly idle (19 and 4.6 MB per layer against ~30 MB on the others); with 128 every engine carries 24-28 MB. The PSUM accumulation order of the down projection changes, so outputs are close but not bit-identical | `NEMOTRON_H_MOE_PREFILL_WTILE` (`128` default since T14d, `exact` = 116-row tiles; ACC kernel only). Unit device profile (T=2048, 2 layers): kernel window per layer 3666 -> 3610 us and 3983 -> 3898 us, weight-queue active time 2214 -> 2014 us. Full model, 2,048-token chunk 219.0 -> 215.7 ms (MoE 139.5 -> 136.6); gate passes; device 0 TTFT -1.0..-2.1% on agent_8k and long_32k over 2 same-session pairs, decode NEFFs unchanged (optimization step 13) |
| MoE experts, prefill, deferred read-modify-write (T12e) | same kernel. Each tile's read-modify-write of the float32 output plane is issued at the start of the next tile of the same expert (or after the expert's dynamic loop), so the all-engine barrier that ends every dynamic-loop iteration no longer waits for that iteration's RMW DMA (`NEMOTRON_H_MOE_PREFILL_DEFER=1`, default; `0` = previous behaviour). Unit 5.03 -> 4.29 ms per layer at 2,048 tokens and 8.72 -> 7.83 at 4,096, bit-identical (also skewed and adversarial routing); full-model 4,096-token chunk 390.3 -> 374.4 ms. Debug knobs from the same task (all off by default): `NEMOTRON_H_MOE_PREFILL_TILE0_LOOPED`, `_CLAMP1`, `_TILE0_LB1`, `_TILE0_LAYERS` and `NEMOTRON_H_DEBUG_EXIT_LAYER` (hidden states after layer L to the LM head) | |
| MoE experts, prefill, next-tile prefetch (R2-03) | same kernel. In chunks of at least 4,096 tokens (`NEMOTRON_H_MOE_PREFILL_TILE_PREFETCH_MIN_T`), each tile transposes its kept x rows and then issues the next tile's slot-index loads, routing-weight gather and x gather into kernel-level buffers, so they run under its matmuls (the all-engine barrier that ends every dynamic-loop iteration had serialized them); the straight-line first tile of each expert loads its own rows and prefetches tile 1; the dispatch adds one pad tile so the last prefetch stays in bounds | `NEMOTRON_H_MOE_PREFILL_TILE_PREFETCH` (default 1 since R2-03; `0` = previous behaviour). Unit (T=4096, random weights): 7.26 -> 6.67 ms per layer, skewed routing 7.58 -> 7.14, bit-identical (at 2,048 tokens +8%, hence the 4,096 minimum). Full model with the Mamba prefill kernel: 4,096-token chunk 297.9 -> 287.7 ms (MoE 208.4 -> 197.0); gate details identical; device-0 agent_8k TTFT -4.3% (c1) / -3.8% (c32) |
| MoE experts, prefill, pad-row gathers skipped (R2-03b) | same kernel. Pad slots (partial last tiles, the max-of-pair padding of the expert split) carry the out-of-bounds token row T, and the x-row gather skips them (`oob_mode=skip`), so they move no bytes; every op of a tile is row-local and a pad row's read-modify-write was already skipped, so real rows are bit-identical. With the bench prompts' routing ~31% of a 4,096-token chunk's slots are pads. BF16 only (the FP8 kernel and dispatch are unchanged). `NEMOTRON_H_MOE_PREFILL_PADSKIP=1` (default; `0` = previous gather). Not kept, default off: `NEMOTRON_H_MOE_PREFILL_UPT=dma[N]` / `_XT=dma[N]` (up-weight / token transposes as DMA transposes: slower, the kernel is DMA-bound), `NEMOTRON_H_MOE_PREFILL_TILES_SL=2` (more straight-line tiles: slower with real routing) | Unit: -2.3% / -2.4% per MoE layer at 4,096 / 2,048 tokens (real routing). Full model: 4,096-token chunk 287.1 -> 284.5 ms, 2,048-token chunk 156.8 -> 153.6 ms. Device 0: TTFT p50 -0.7..-3.0% on the Must rows, TPOT unchanged, gate passes |
| MoE experts, prefill, slot pointer carried across experts (R2-03d) | same kernel. In chunks of >= 4,096 tokens (the ones that use TILE_PREFETCH) the dispatch lays each core's slot ranges out contiguously in that core's processing order instead of by expert id, so the last tile prefetch of an expert loads the next expert's first tile (before, it gathered 0.7 MB of unrelated rows that the loop barrier waited for), and the kernel carries the slot pointer and prefetched rows across experts. Same experts, order and arithmetic per core: bit-identical. BF16 only. `NEMOTRON_H_MOE_PREFILL_XCARRY=1` (default only with `NEMOTRON_H_MAMBA_PREFILL_INPROJ=torch`; `0` = slot ranges by expert id, the default with the in_proj kernel since R2-03c5: with `fused`/`nki` the carried layout makes neuronx-cc spill 2.7-3.6 GB more per core in 8 MoE blocks, +29 ms per 4,096-token chunk; an explicit setting wins). Not kept, default off: `NEMOTRON_H_MOE_PREFILL_FIRSTW=1` (first-touch plane writes instead of zeroing + RMW: slower and not bit-identical, the plain and RMW scatters to a row are not ordered), `NEMOTRON_H_MOE_PREFILL_NPRE=1` (trip counts in one DMA: no gain) | Unit (real routing): 5.95 -> 5.65 ms per MoE layer at 4,096 tokens. Full model: 4,096-token chunk 283.9 -> 278.9 ms. Device 0: TTFT p50 agent_8k / long_32k -1.7..-2.1%, TPOT unchanged, gate details identical |
| Shared expert | torch | |
| Sampler | `vllm_neuron.nn.sampler.Sampler` | |

### MoE kernels: can the NKI Library take non-gated squared-ReLU experts?

Checked in the installed `nkilib` (DLAMI 2.33, `site-packages/nkilib`):

* `nkilib.core.utils.common_types.ActFnType` has `SquaredReLU = 5`
  (`kernel_helpers.apply_activation` implements it as relu then square).
* **CTE (prefill)**: `nkilib/core/moe/moe_cte` supports non-gated experts through
  a `skip_gate_proj` flag (`moe_cte.py`, `bwmm_shard_on_I.py`, `moe_cte_utils.py`
  apply `SquaredReLU` to the up projection alone when `skip_gate_proj` is set).
  The weight layout is still `gate_up [E, H, 2, I]`.
* **TKG (decode)**: `nkilib/core/moe/moe_tkg/mlp_tkg_gate_up_projection*.py`
  and `mlp_parameters.py` also carry `skip_gate_proj`.
* **vLLM Neuron wrappers**: `vllm_neuron/functional/moe/moe_cte.py` does not pass
  `skip_gate_proj` through, and `moe_block_tkg.py` maps only SiLU/GELU/Swish
  (`hidden_act_fn`). So the kernels exist, but wiring them in needs a small
  wrapper change.
* Without touching the wrappers there is an exact workaround:
  `relu(u) * u == relu(u)^2` for every `u`, so a **gated** kernel with
  `ActFnType.ReLU` and the gate weights set equal to the up weights computes the
  non-gated squared-ReLU expert exactly, at the cost of reading the up weights
  twice.
* Measured (T09b, unit test, 8 MoE layers at batch 1, TP=4 rank-0 shapes, ms per layer
  including per-call overhead): dense 1.143, T14 `moe_topk` 0.309, `moe_tkg` gated-ReLU with
  gate = up 0.306, `moe_tkg` with `skip_gate_proj` + SquaredReLU (patched) 0.231, `moe_tkg`
  all-expert mode 2.41. At batch 32: dense 1.33, selective 1.78, all-expert 2.38.

## Feature status

| Feature | Status | Notes |
|---|---|---|
| TP=4 on one Trainium2 device (LNC=2) | ✅ | the tested configuration |
| TP=1 (CPU mode) | ✅ | `VLLM_NEURON_CPU_MODE=1`, matches HF fp32 to 5e-5 on a 4-layer checkpoint |
| Sequence parallelism (prefill) | ✅ | `VLLM_NEURON_NEMOTRON_H_DISABLE_SP=1` all-reduces instead |
| Chunked prefill | ✅ | Mamba conv/SSM state and KV prefix carried across chunks; set `kv_segment_size_buckets` |
| Continuous batching | ✅ | batch-invariance gate passes |
| Prefix caching (Mamba `align` mode) | ✅ opt-in | state bound into the shared KV pool (`VLLM_NEURON_STATE_KV_ALIAS`); cache-hit gate 7/7 |
| NKI Mamba-2 | ✅ | decode: `kernels/mamba_decode.py`; prefill: `kernels/ssd_prefill.py` with an NKI `in_proj` |
| NKI MoE experts | ✅ | decode batch 1: `kernels/moe_topk.py` + `kernels/router_top8.py`; decode buckets 8-16: `kernels/moe_decode_routed.py`; 32-128: dense torch (the routed kernel was not faster there); prefill: `kernels/moe_prefill.py` |
| FP8 weights (routed experts) | ✅ variant, `NEMOTRON_H_FP8=1` (default off) | see "FP8 weights variant" below |
| EP | ❌ | not needed at TP=4 (the model fits one device) |
| FP8 KV | ❌ | |
| MTP speculative decoding | ❌ | `mtp.*` weights skipped (an experimental branch gave greedy outputs that differ from decoding without MTP) |
| `mlp` (dense) block type | ❌ | raises; not used by 30B-A3B |

## FP8 weights variant (R2-05, `NEMOTRON_H_FP8=1`, default off)

A separate variant, not part of the BF16 headline. `NEMOTRON_H_FP8=1` quantizes the **routed experts' up and down
weights** at load time from the BF16 checkpoint. Activations stay bf16 and every matmul accumulates in fp32 PSUM.
Router, norms, embeddings, LM head, attention, Mamba and the shared expert stay as they were (bf16 / fp32).

* **Format.** Stored as torch `float8_e4m3fn` with every value clamped to +-240, which is bit-identical to trn2's
  Tensor/Vector Engine format `nl.float8_e4m3` (IEEE-style e4m3, bias 7, max 240). NKI maps torch `float8_e4m3fn`
  inputs to `nl.float8_e4m3` on trn2 only with `UNSAFE_FP8FNCAST=1`; the flag sets it (`os.environ.setdefault`).
  trn2 has no OCP `float8_e4m3fn` (max 448) in the engines (gen4+ only), so values above 240 would decode as inf/NaN.
* **Scales.** One fp32 scale per row of the stored `[E * I, H]` matrices, `s = row amax / 240` (round to nearest even):
  per output channel of `up_proj`, per input channel of `down_proj`. Both index the intermediate channel `i`, and
  `relu(s * u)^2 = s^2 relu(u)^2` for `s > 0`, so each expert is
  `sum_i relu((Wq_up x)_i)^2 * c_i * Wq_down[i, :]` with one vector `c_i = s_up_i^2 * s_down_i` (`experts_c`,
  `[E * I + 128, 1]`). `c` scales the activation in all three paths and never touches a weight tile. Per-output-channel
  scales on `down` would not fold this way, because the batch-1 kernel accumulates the 6 experts in one PSUM tile and
  the dense path sums all experts in one matmul.
* **Layouts and kernels.**
  | Path | Kernel | FP8 change |
  |---|---|---|
  | decode, batch 1 | `kernels/moe_topk.py` (`_moe_topk_kernel_fp8`) | the same kernel on fp8 row tiles: Vector Engine dot products read fp8 `up`, the Tensor Engine reads fp8 `down` as the moving operand (bf16 x fp8), `a *= c` per row |
  | decode, batches 8-128 | `kernels/moe_fp8.py` `moe_fp8_dense` (new) | replaces the dense all-experts torch matmuls. A second fp8 copy of `up` stored transposed, `[E * 128, nh * I]` (`experts_up_t`), is one DMA per expert; `uT = upT^T @ xT` (fp8 weight stationary, I on partitions), `aT = relu(uT)^2 * c * w`, `y += aT^T @ down` (one PSUM bank per 512 columns). Experts split over the 2 physical cores |
  | decode, routed buckets (`NEMOTRON_H_MOE_ROUTED_BUCKETS`, default 8 and 16) | `kernels/moe_decode_routed.py` `moe_routed_fp8_kernel` (R2-05b, `NEMOTRON_H_FP8_ROUTED=1`, default on with `NEMOTRON_H_FP8=1`) | the R2-01 compact expert list, parity split and dynamic loop, on the fp8 tiles of `moe_fp8_dense`: one DMA of the expert's `upT` from `experts_up_t` (no Tensor-Engine transposes), `uT = upT^T @ xT` (fp8 stationary), `aT = relu(uT)^2 * c`, `y += w[:, e] * (aT^T @ down)` (routing weight applied per PSUM row). 4 load buffers (`NEMOTRON_H_FP8_ROUTED_BUFS`, 2/3/4). `NEMOTRON_H_FP8_ROUTED=0` sends these buckets to `moe_fp8_dense` (the R2-05 merge behaviour) |
  | prefill | `kernels/moe_prefill.py` (`moe_prefill_kernel_esplit_acc_fp8`) | `upT` DMA'd from `experts_up_t` (no Tensor-Engine transposes; an fp8 PE transpose needs a stride-2 fp8 PSUM destination), fp8 moving operands, `c` applied when `aT` leaves PSUM |
  Casting the fp8 weights to bf16 in torch (`F.linear(x, w.to(bf16))`) is slower than bf16 (unit, batch 32: 2.27 vs
  1.43 ms per layer), because neuronx-cc does not fuse the cast into the matmul load.
* **Memory.** Weights per rank at TP=4: 11.32 GiB with `NEMOTRON_H_FP8=1` (two fp8 copies of `up`, one of `down`)
  vs 14.74 GiB BF16. Each decode step reads one `up` copy, so routed-expert bytes per step are halved.
* **Sub-options.** `NEMOTRON_H_FP8_SHARED=1` (default 0) also puts the shared expert into the decode kernels as 2
  pseudo-experts (weight 1); prefill keeps the bf16 shared expert. It doubled the per-layer quantization error in the
  unit test (rel. L2 vs bf16 1.2% -> 2.3%) for no c=1 gain on the full model (agent_8k c1 TPOT 6.51 -> 6.48 ms), so it
  stays off. `NEMOTRON_H_FP8_TOPK=t` runs batch 1 through `moe_fp8_topk` (weights-stationary, transposed `up`), slower
  than the default at batch 1 in the unit test.
* **Routed kernel options (R2-05c).** `NEMOTRON_H_FP8_ROUTED_V2` (comma list; default `skip,peel`; `none` or empty =
  the R2-05b kernel). `skip`: pad entries of the expert list are not loaded (`oob_mode=skip`); `peel`: the first loop
  iteration runs before the loop so its computes overlap the prologue loads. Together 4-8% less device time per MoE
  layer at T=8/16 and bit-identical outputs. `wfold`: the routing weight is applied before the bf16 rounding of the
  activation, as in `moe_fp8_dense` and the batch-1 kernel (the R2-05b kernel's rounding differs from the batch-1
  kernel's 4x more, which failed the batch-invariance gate at bucket 32); with
  `NEMOTRON_H_MOE_ROUTED_BUCKETS=8,16,32` the gate passes and bucket 32 runs the routed kernel. Other options
  (`early`, `hw`, `uhw`, `dhw`, `actrelu`, 3/6/8 buffers) were slower or equal; `psum` does not compile.
* **Results.** Correctness gate, GSM8K, tool calling, needle and the Must rows with the flag on and off are in the write-up's FP8 section.

