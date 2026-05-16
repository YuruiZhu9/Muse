"""Grid search over semantic projection hyperparameters to maximize LLM gain."""
import json, sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

# Configurations to sweep
SWEEPS = [
    # (llm_proj_dim, semantic_num_heads, semantic_dropout, learning_rate, label)
    (128, 4, 0.0, 1e-3, "proj128_h4"),
    (128, 8, 0.0, 1e-3, "proj128_h8"),
    (128, 8, 0.1, 1e-3, "proj128_h8_d01"),
    (256, 4, 0.0, 1e-3, "proj256_h4"),
    (256, 8, 0.1, 1e-3, "proj256_h8_d01"),  # current best
    (256, 8, 0.2, 5e-4, "proj256_h8_d02_lr5e4"),
    (512, 8, 0.1, 1e-3, "proj512_h8_d01"),
    (512, 8, 0.2, 5e-4, "proj512_h8_d02_lr5e4"),
    (512, 16, 0.2, 5e-4, "proj512_h16_d02_lr5e4"),
]

BEST_FILE = Path("LLM_part/sweep_best.json")


def run_one(proj_dim, num_heads, dropout, lr, label):
    import subprocess
    cmd = [
        sys.executable, "run_kuairec_offline_experiment.py",
        "--user-inference-file", "LLM_part/user_inferences_big_train_phase1_local_compact.jsonl",
        "--cutoff-mode", "anchor_date",
        "--max-eval-users", "100",
        "--experiment", "compare_all",
        "--text-encoder", "hash",
        "--top-k", "10",
        "--epochs", "15",
        "--early-stop-patience", "5",
        "--learning-rate", str(lr),
        "--output-file", f"LLM_part/sweep_{label}.json",
    ]
    env = {
        **__import__("os").environ,
        "KUAIREC_SEMANTIC_PROJ_DIM": str(proj_dim),
        "KUAIREC_SEMANTIC_NUM_HEADS": str(num_heads),
        "KUAIREC_SEMANTIC_DROPOUT": str(dropout),
    }
    print(f"\n{'='*60}")
    print(f"[sweep] {label}: proj={proj_dim} heads={num_heads} drop={dropout} lr={lr}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[sweep] {label}: FAILED (exit {result.returncode})")
        return None
    print(f"[sweep] {label}: done in {elapsed:.0f}s")
    return f"LLM_part/sweep_{label}.json"


def evaluate(result_file, label):
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    din = data["experiments"]["din_only"]["metrics"]
    llm = data["experiments"]["llm_full"]["metrics"]
    scores = {}
    for mk in ["ctr_auc", "ctr_gauc", "cvr_auc", "cvr_gauc", "ndcg@10", "hit_rate@10", "mrr@10"]:
        d, l = din.get(mk, 0), llm.get(mk, 0)
        delta = (l - d) / max(abs(d), 0.01) * 100
        scores[mk] = {"din": d, "llm": l, "delta_pct": round(delta, 2)}
    # Composite score: average delta across GAUC + NDCG + MRR
    composite = np.mean([scores[m]["delta_pct"] for m in ["ctr_gauc", "cvr_gauc", "ndcg@10", "mrr@10"]])
    scores["_composite"] = round(float(composite), 2)
    scores["_label"] = label
    return scores


def main():
    results = []
    for proj_dim, num_heads, dropout, lr, label in SWEEPS:
        result_file = run_one(proj_dim, num_heads, dropout, lr, label)
        if result_file is None:
            continue
        scores = evaluate(result_file, label)
        results.append(scores)
        print(f"  composite={scores['_composite']} | "
              f"GAUC_ctr={scores['ctr_gauc']['delta_pct']}% "
              f"GAUC_cvr={scores['cvr_gauc']['delta_pct']}% "
              f"NDCG={scores['ndcg@10']['delta_pct']}% "
              f"MRR={scores['mrr@10']['delta_pct']}%")

    results.sort(key=lambda r: r["_composite"], reverse=True)
    print(f"\n{'='*60}")
    print("RANKED RESULTS (by composite score):")
    for i, r in enumerate(results):
        print(f"  #{i+1} {r['_label']}: composite={r['_composite']} "
              f"| GAUC_ctr={r['ctr_gauc']['delta_pct']}% "
              f"NDCG={r['ndcg@10']['delta_pct']}% "
              f"MRR={r['mrr@10']['delta_pct']}%")

    if results:
        best = results[0]
        print(f"\nBEST: {best['_label']} (composite={best['_composite']})")
        BEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_FILE, "w", encoding="utf-8") as f:
            json.dump({"best_label": best["_label"], "results": results}, f, ensure_ascii=False, indent=2)
        print(f"Saved to {BEST_FILE}")


if __name__ == "__main__":
    main()
