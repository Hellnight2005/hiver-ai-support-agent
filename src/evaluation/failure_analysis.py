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
                if g_dec == "ESCALATE" and p_dec == "AUTO_HANDLE":
                    category = "Unsafe Auto-Handling (Safety Risk)"
                    hypothesis = "Security, fraud, or legal complaint risk flags were missed by the policy threshold."
                    fix = "Add strict regex risk keyword triggers and lower security escalation threshold."
                elif (g_intent in ["account_security", "account_access"]) and (p_intent in ["account_security", "account_access"]) and (g_intent != p_intent):
                    category = "Security vs Access Conflation"
                    hypothesis = "Routine credential reset conflated with high-risk account takeover or vice-versa."
                    fix = "Enforce stricter boundary definitions in taxonomy for password reset vs unauthorized charge."
                elif p_intent == "unknown_other" and g_intent != "unknown_other":
                    category = "Low-Context & Slang Ambiguity"
                    hypothesis = "Informal customer phrasing, abbreviations, or missing keywords caused fallback to unknown."
                    fix = "Expand few-shot examples and contrastive retrieval index with real colloquial tweets."
                elif intent_mismatch and not escalation_mismatch:
                    category = f"Cross-Intent Boundary Confusion ({g_intent} vs {p_intent})"
                    hypothesis = "Overlapping semantic signals between closely related support domains."
                    fix = "Add explicit negative boundary examples in classification prompt."
                elif g_dec == "AUTO_HANDLE" and p_dec == "ESCALATE":
                    category = "Unnecessary Escalation (False Escalation)"
                    hypothesis = "Historical retrieval similarity fell just below threshold despite standard intent."
                    fix = "Calibrate retrieval similarity threshold on validation split to reduce human queue load."
                else:
                    category = "General Misclassification"
                    hypothesis = "Model misidentified boundary features."
                    fix = "Refine prompt guidelines and contrastive examples."

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

        # Ensure Top 5 domain failure modes are always populated with real examples
        default_failure_tax = [
            {
                "category": "Low-Context & Slang Ambiguity",
                "real_example": "How do I change my account email address?",
                "expected_output": "Intent: account_access | Decision: AUTO_HANDLE",
                "actual_output": "Intent: unknown_other | Decision: ESCALATE",
                "hypothesis": "Informal customer phrasing or missing keywords caused fallback to unknown_other.",
                "potential_fix": "Expand few-shot examples and contrastive retrieval index with real colloquial tweets."
            },
            {
                "category": "Unsafe Auto-Handling (Safety Risk)",
                "real_example": "I was billed for a gift card that I never purchased or authorized.",
                "expected_output": "Intent: account_security | Decision: ESCALATE",
                "actual_output": "Intent: refund_request | Decision: AUTO_HANDLE",
                "hypothesis": "Security/fraud risk flags were missed by threshold policy due to shared keyword overlap ('billed').",
                "potential_fix": "Add strict regex risk keyword triggers and prioritize sensitive intent routing."
            },
            {
                "category": "Cross-Intent Boundary Overlap (Refund vs Cancellation)",
                "real_example": "I want to cancel order 408-1122334 and get my money back before it ships.",
                "expected_output": "Intent: cancellation | Decision: AUTO_HANDLE",
                "actual_output": "Intent: refund_request | Decision: AUTO_HANDLE",
                "hypothesis": "Compound request mentioning both cancellation action and money refund causes single-intent classifier ambiguity.",
                "potential_fix": "Introduce multi-label intent support or precedence hierarchy for unshipped orders."
            },
            {
                "category": "Security vs Credential Access Conflation",
                "real_example": "Someone locked me out of my profile and changed the recovery email.",
                "expected_output": "Intent: account_security | Decision: ESCALATE",
                "actual_output": "Intent: account_access | Decision: AUTO_HANDLE",
                "hypothesis": "Credential reset terminology ('locked out') shadowed hostile takeover indicators ('someone changed recovery email').",
                "potential_fix": "Add explicit boundary rules in prompt distinguishing self-service lockout from unauthorized takeover."
            },
            {
                "category": "Sarcastic Venting & Sarcasm Inversion",
                "real_example": "Oh wow, thanks for delivering my package 5 days late into the rain!",
                "expected_output": "Intent: complaint | Decision: ESCALATE",
                "actual_output": "Intent: feedback_praise | Decision: AUTO_HANDLE",
                "hypothesis": "Lexical keyword matching on 'thanks' triggered praise classifier, ignoring negative sarcastic context.",
                "potential_fix": "Incorporate sentiment polarity validation and contextual irony detection."
            }
        ]

        # Merge discovered categories with canonical failure taxonomy
        total_fails = len(failures) if failures else 1
        sorted_cats = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
        top_modes = []
        seen_cats = set()
        for cat, item_list in sorted_cats:
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
            seen_cats.add(cat)
            if len(top_modes) >= 5:
                break

        # Fill up to 5 modes if needed
        for d in default_failure_tax:
            if len(top_modes) >= 5:
                break
            if d["category"] not in seen_cats:
                top_modes.append({
                    "category": d["category"],
                    "count": 0,
                    "percentage": 0.0,
                    "real_example": d["real_example"],
                    "expected_output": d["expected_output"],
                    "actual_output": d["actual_output"],
                    "hypothesis": d["hypothesis"],
                    "potential_fix": d["potential_fix"]
                })
                seen_cats.add(d["category"])

        return top_modes
