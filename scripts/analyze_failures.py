import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.failure_analysis import FailureAnalyzer


def main():
    print("==================================================")
    print(" 11. FAILURE ANALYSIS ENGINE")
    print("==================================================")

    eval_file = Path(settings.paths.evaluation_results)
    if not eval_file.exists():
        print(f"Error: {eval_file} not found. Please run scripts/run_evaluation.py first.")
        sys.exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    agent_records = results.get("AI Agent", {}).get("records", [])
    analyzer = FailureAnalyzer()
    top_failures = analyzer.analyze(agent_records)

    out_md = Path("reports/failure_analysis.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md_content = "# Failure Mode Analysis Report\n\n"
    md_content += "Derived top failure modes from AI Customer Support Agent evaluation.\n\n"

    for idx, f in enumerate(top_failures, 1):
        md_content += f"## Failure Mode #{idx}: {f['category']}\n"
        md_content += f"- **Count**: {f['count']} ({f['percentage']}% of failures)\n"
        md_content += f"- **Real Customer Example**: \"{f['real_example']}\"\n"
        md_content += f"- **Expected Output**: `{f['expected_output']}`\n"
        md_content += f"- **Actual Output**: `{f['actual_output']}`\n"
        md_content += f"- **Hypothesis**: {f['hypothesis']}\n"
        md_content += f"- **Potential Fix**: {f['potential_fix']}\n\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Failure analysis complete! Report generated at: {out_md}\n")
    print(f"Discovered {len(top_failures)} top failure categories:")
    for f in top_failures:
        print(f"  - [{f['count']} cases / {f['percentage']}%] {f['category']}")


if __name__ == "__main__":
    main()
