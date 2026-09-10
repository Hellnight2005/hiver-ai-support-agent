from typing import List, Dict, Any


class FailureAnalyzer:
    def analyze(self, evaluation_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failures = []
        for r in evaluation_records:
            g_intent = r.get("gold_intent", "")
            p_intent = r.get("pred_intent", "")
            g_dec = r.get("gold_escalation", "").upper()
            p_dec = r.get("pred_escalation", "").upper()

            intent_mismatch = (g_intent != p_intent)
            escalation_mismatch = (g_dec != p_dec)

            if intent_mismatch or escalation_mismatch:
                category = "General Misclassification"
                hypothesis = "Model misidentified boundary features."
                fix = "Refine prompt guidelines and contrastive examples."

                if g_dec == "ESCALATE" and p_dec == "AUTO_HANDLE":
                    category = "Unsafe Auto-Handling Failure"
                    hypothesis = "Security/fraud risk flags were missed by threshold policy."
                    fix = "Add strict keyword triggers and lower security escalation threshold."
                elif g_dec == "AUTO_HANDLE" and p_dec == "ESCALATE":
                    category = "Unnecessary Escalation Failure"
                    hypothesis = "Retrieval confidence fell just below threshold despite valid intent."
                    fix = "Tune retrieval similarity threshold on validation data."
                elif "refund" in g_intent and "charge" in p_intent:
                    category = "Ambiguous Billing Language"
                    hypothesis = "Transaction charge terminology shared across refund and billing intents."
                    fix = "Add contrastive taxonomy examples between refund_request and transaction_problem."
                elif p_intent == "unknown_other":
                    category = "High Ambiguity / Unknown Intent"
                    hypothesis = "Input message contained non-standard syntax or abbreviations."
                    fix = "Expand training retrieval corpus with informal customer syntax."

                failures.append({
                    "id": r.get("id", "unk"),
                    "customer_message": r.get("message", ""),
                    "gold_intent": g_intent,
                    "pred_intent": p_intent,
                    "gold_escalation": g_dec,
                    "pred_escalation": p_dec,
                    "category": category,
                    "hypothesis": hypothesis,
                    "potential_fix": fix
                })

        # Group by category
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for f in failures:
            cat = f["category"]
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(f)

        top_modes = []
        total_fails = len(failures) if failures else 1
        sorted_cats = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

        for cat, item_list in sorted_cats[:5]:
            sample = item_list[0]
            top_modes.append({
                "category": cat,
                "count": len(item_list),
                "percentage": round((len(item_list) / total_fails) * 100, 1),
                "real_example": sample["customer_message"],
                "expected_output": f"Intent: {sample['gold_intent']} | Decision: {sample['gold_escalation']}",
                "actual_output": f"Intent: {sample['pred_intent']} | Decision: {sample['pred_escalation']}",
                "hypothesis": sample["hypothesis"],
                "potential_fix": sample["potential_fix"]
            })

        return top_modes
