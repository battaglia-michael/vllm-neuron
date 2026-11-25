#!/bin/bash
# Run the CPU-mode numerics matrix (fp32/bf16 x torch-chunk/segmented chunked prefill).
# Usage: run_cpu_numerics.sh <ckpt_fp32> <ckpt_bf16> <out_dir> <log_dir>
set -u
CK32=$1; CK16=$2; OUT=$3; LOG=$4
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$VLLM_VENV/bin/activate"
export VLLM_NEURON_CPU_MODE=1 NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.002 OMP_NUM_THREADS=${OMP_NUM_THREADS:-40}
mkdir -p "$OUT" "$LOG"
run() {  # name ckpt dtype extra...
  local name=$1 ck=$2 dt=$3; shift 3
  python "$HERE/test/integration/cpu_numerics.py" --ckpt "$ck" --dtype "$dt" --out "$OUT/$name.json" "$@" > "$LOG/$name.log" 2>&1
  local rc=$?; echo "$name exit=$rc"; return $rc
}
run fp32-chunk256 "$CK32" float32 --max-model-len 1024 --max-num-batched-tokens 256 --long-tokens 700 & p1=$!
run fp32-seg512   "$CK32" float32 --max-model-len 2048 --max-num-batched-tokens 512 --long-tokens 1300 --kv-segment 512 & p2=$!
run bf16-chunk256 "$CK16" bfloat16 --max-model-len 1024 --max-num-batched-tokens 256 --long-tokens 700 & p3=$!
run bf16-seg512   "$CK16" bfloat16 --max-model-len 2048 --max-num-batched-tokens 512 --long-tokens 1300 --kv-segment 512 & p4=$!
fail=0
for p in $p1 $p2 $p3 $p4; do wait $p || fail=1; done
echo "overall: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
