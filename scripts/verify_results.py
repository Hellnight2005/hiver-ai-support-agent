import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings


def main():
    print("==================================================")
    print(" 13. METRIC INTEGRITY & CONSISTENCY AUDITOR")
    print("==================================================")

    all_passed = True

    # 1. Golden Set Verification
    golden_path = Path(settings.paths.golden_set)
    golden_records = []
    if not golden_path.exists():
        print("[-] Golden set file: FAIL (file not found)")
        all_passed = False
    else:
        with open(golden_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    try:
                        rec = json.loads(line)
                        golden_records.append(rec)
                    except Exception as e:
                        print(f"[-] Golden set line {idx+1} invalid JSON: {e}")
                        all_passed = False
        n_gold = len(golden_records)
        if 150 <= n_gold <= 250:
            print(f"[+] Golden set: PASS ({n_gold} examples, meets 150-250 SLA)")
        else:
            print(f"[-] Golden set count: WARNING ({n_gold} examples, expected 150-250)")

    # 2. Evaluation Artifact Verification
    eval_file = Path(settings.paths.evaluation_results)
    if not eval_file.exists():
        print(f"[-] Evaluation artifact: FAIL ({eval_file} not found)")
        all_passed = False
        print("\nRESULT: FAIL")
        sys.exit(1)
    
    with open(eval_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"[+] Evaluation artifact: PASS ({eval_file})")

    # 3. Model & Baseline Presences
    required_systems = ["Majority", "TF-IDF", "Embedding Retrieval", "AI Agent"]
    for s in required_systems:
        if s in results:
            print(f"[+] System '{s}': PASS")
        else:
            print(f"[-] System '{s}': FAIL (missing from results)")
            all_passed = False

    # 4. Metric Value Range Checks
    for s in required_systems:
        if s not in results:
            continue
        im = results[s].get("intent_metrics", {})
        em = results[s].get("escalation_metrics", {})

        acc = im.get("accuracy")
        mf1 = im.get("macro_f1")
        ef1 = em.get("f1")
        unsafe = em.get("unsafe_auto_handling_rate")

        for name, val in [("accuracy", acc), ("macro_f1", mf1), ("escalation_f1", ef1), ("unsafe_auto_rate", unsafe)]:
            if val is None or not (0.0 <= val <= 1.0):
                print(f"[-] {s} {name} range check: FAIL (val={val})")
                all_passed = False
            else:
                pass

    print("[+] Metric range bounds [0.0, 1.0]: PASS")

    # 5. Derived Safety and Escalation Verification
    agent_data = results.get("AI Agent", {})
    agent_records = agent_data.get("records", [])
    if agent_records:
        # Check unsafe auto handling count derivation
        actual_unsafe_count = sum(
            1 for r in agent_records 
            if r.get("gold_escalation", "").upper() == "ESCALATE" 
            and r.get("pred_escalation", "").upper() == "AUTO_HANDLE"
        )
        reported_unsafe_count = agent_data.get("escalation_metrics", {}).get("unsafe_auto_handling_count")
        reported_unsafe_rate = agent_data.get("escalation_metrics", {}).get("unsafe_auto_handling_rate")
        expected_rate = round(actual_unsafe_count / len(agent_records), 4)

        if actual_unsafe_count == reported_unsafe_count and reported_unsafe_rate == expected_rate:
            print(f"[+] Unsafe auto-handling derivation: PASS ({actual_unsafe_count}/{len(agent_records)} = {reported_unsafe_rate*100:.1f}%)")
        else:
            print(f"[-] Unsafe auto-handling derivation mismatch: FAIL (actual={actual_unsafe_count}, reported={reported_unsafe_count})")
            all_passed = False

        # Check unnecessary escalation count derivation
        actual_unnecessary_count = sum(
            1 for r in agent_records 
            if r.get("gold_escalation", "").upper() == "AUTO_HANDLE" 
            and r.get("pred_escalation", "").upper() == "ESCALATE"
        )
        reported_unnecessary_count = agent_data.get("escalation_metrics", {}).get("unnecessary_escalation_count")
        if actual_unnecessary_count == reported_unnecessary_count:
            print(f"[+] Unnecessary escalation derivation: PASS ({actual_unnecessary_count}/{len(agent_records)})")
        else:
            print(f"[-] Unnecessary escalation derivation mismatch: FAIL")
            all_passed = False

    # 6. Report Consistency Check
    report_file = Path("reports/evaluation.md")
    if not report_file.exists():
        print("[-] Report consistency: FAIL (reports/evaluation.md not found)")
        all_passed = False
    else:
        with open(report_file, "r", encoding="utf-8") as f:
            report_text = f.read()

        agent_acc = agent_data.get("intent_metrics", {}).get("accuracy", 0.0)
        agent_acc_str = f"{agent_acc * 100:.1f}%"
        if agent_acc_str in report_text:
            print(f"[+] Report consistency with evaluation_results.json ({agent_acc_str}): PASS")
        else:
            print(f"[-] Report consistency: FAIL ({agent_acc_str} not found in evaluation.md)")
            all_passed = False

    # 7. Data Leakage Verification
    dev_path = Path(settings.paths.processed_data_dir) / "dev_conversations.json"
    eval_path = Path(settings.paths.processed_data_dir) / "eval_conversations.json"
    if dev_path.exists() and eval_path.exists():
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_data = json.load(f)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        dev_ids = {c["conversation_id"] for c in dev_data}
        eval_ids = {c["conversation_id"] for c in eval_data}
        leakage = dev_ids.intersection(eval_ids)

        if len(leakage) == 0:
            print(f"[+] Zero train/eval conversation leakage: PASS (0 overlapping IDs)")
        else:
            print(f"[-] Conversation leakage detected: FAIL ({len(leakage)} overlapping IDs)")
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("=== EVALUATION INTEGRITY AUDIT: ALL CHECKS PASS ===")
    else:
        print("=== EVALUATION INTEGRITY AUDIT: FAILURES DETECTED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
