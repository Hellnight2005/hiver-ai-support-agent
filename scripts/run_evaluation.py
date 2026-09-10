import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.evaluator import SystemEvaluator


def main():
    print("==================================================")
    print(" 9. RUN EVALUATION HARNESS")
    print("==================================================")

    evaluator = SystemEvaluator()
    print("Running evaluation across Majority, TF-IDF, Embedding Retrieval, and AI Agent...")
    results = evaluator.run_all()

    # Save to artifacts/evaluation_results.json
    out_file = Path(settings.paths.evaluation_results)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete! Results saved to: {out_file}\n")
    print("=" * 85)
    print(f"{'System':<22} | {'Intent Acc':<10} | {'Macro F1':<9} | {'Esc F1':<8} | {'Unsafe Auto':<11} | {'Reply Score':<11}")
    print("=" * 85)

    for sys_name, res in results.items():
        im = res["intent_metrics"]
        em = res["escalation_metrics"]
        score = res.get("avg_reply_score")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"{sys_name:<22} | {im.get('accuracy', 0):<10.4f} | {im.get('macro_f1', 0):<9.4f} | {em.get('f1', 0):<8.4f} | {em.get('unsafe_auto_handling_rate', 0):<11.4f} | {score_str:<11}")
    print("=" * 85)


if __name__ == "__main__":
    main()
