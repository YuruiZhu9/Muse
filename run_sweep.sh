#!/bin/bash
# Grid search for best semantic projection config
set -e
cd "C:/Users/53025/MuseRecSys"
INF="LLM_part/user_inferences_big_train_phase1_local_compact.jsonl"
BASE="--user-inference-file $INF --cutoff-mode anchor_date --max-eval-users 100 --experiment compare_all --text-encoder hash --top-k 10 --epochs 15 --early-stop-patience 5"

SWEEPS=(
  # proj_dim heads dropout lr label
  "64 4 0.0 1e-3 p64_h4"
  "128 4 0.0 1e-3 p128_h4"
  "128 4 0.1 1e-3 p128_h4_d01"
  "128 8 0.0 1e-3 p128_h8"
  "128 8 0.1 1e-3 p128_h8_d01"
  "128 8 0.2 1e-3 p128_h8_d02"
  "192 8 0.1 1e-3 p192_h8_d01"
  "256 8 0.1 1e-3 p256_h8_d01"
  "256 8 0.2 5e-4 p256_h8_d02_lr5e4"
  "256 8 0.2 1e-3 p256_h8_d02"
  "384 8 0.2 5e-4 p384_h8_d02_lr5e4"
  "512 8 0.2 5e-4 p512_h8_d02_lr5e4"
)

for cfg in "${SWEEPS[@]}"; do
  read proj heads drop lr label <<< "$cfg"
  echo ""
  echo "========================================"
  echo "SWEEP: $label (proj=$proj heads=$heads drop=$drop lr=$lr)"
  echo "========================================"
  python run_kuairec_offline_experiment.py $BASE \
    --llm-proj-dim $proj --semantic-num-heads $heads --semantic-dropout $drop \
    --learning-rate $lr \
    --output-file "LLM_part/sweep_${label}.json" 2>&1 | tee "LLM_part/sweep_${label}.log"

  # Quick summary
  python -c "
import json
with open('LLM_part/sweep_${label}.json','r',encoding='utf-8') as f:
    d=json.load(f)
din=d['experiments']['din_only']['metrics']
llm=d['experiments']['llm_full']['metrics']
def pct(a,b): return (b-a)/max(abs(a),0.01)*100
print(f'  [${label}] GAUC_ctr: {din[\"ctr_gauc\"]:.4f}->{llm[\"ctr_gauc\"]:.4f} ({pct(din[\"ctr_gauc\"],llm[\"ctr_gauc\"]):+.1f}%) | NDCG: {pct(din[\"ndcg@10\"],llm[\"ndcg@10\"]):+.1f}% | MRR: {pct(din[\"mrr@10\"],llm[\"mrr@10\"]):+.1f}%')
" 2>/dev/null
done

echo ""
echo "===== ALL SWEEPS COMPLETE ====="
