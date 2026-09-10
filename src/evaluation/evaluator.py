import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import settings
from src.agent.agent import SupportAgent
from src.baselines.majority import MajorityBaseline
from src.baselines.keyword import TFIDFBaseline
from src.baselines.retrieval import EmbeddingRetrievalBaseline
from src.evaluation.metrics import EvaluationMetricsCalculator
from src.evaluation.judge import LLMJudge


class SystemEvaluator:
    def __init__(self, force_mock: bool = False):
        self.force_mock = force_mock
        self.agent = SupportAgent()
        self.majority = MajorityBaseline()
        self.tfidf = TFIDFBaseline()
        self.emb_retrieval = EmbeddingRetrievalBaseline()
        self.judge = LLMJudge()

    def load_golden_set(self, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        path = Path(filepath or settings.paths.golden_set)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset file not found at {path}")
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def run_all(self, golden_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        records = golden_records or self.load_golden_set()
        
        # Fit baselines on development set conversations ONLY
        dev_file = Path(settings.paths.processed_data_dir) / "dev_conversations.json"
        dev_items = []
        if dev_file.exists():
            with open(dev_file, "r", encoding="utf-8") as f:
                dev_data = json.load(f)
            for idx, c in enumerate(dev_data):
                msg = c.get("messages", [{}])[0].get("text", "")
                resp = c.get("messages", [{}, {}])[1].get("text", "Please DM us your details.") if len(c.get("messages", [])) > 1 else "Please DM us."
                # Infer intent from message content instead of defaulting all to one class
                intent = self._infer_dev_intent(msg)
                dev_items.append({
                    "conversation_id": c.get("conversation_id", f"dev_{idx}"),
                    "customer_message": msg,
                    "agent_response": resp,
                    "intent": intent
                })

        if not dev_items:
            dev_items = [
                {"conversation_id": "d1", "customer_message": "Where is my package?", "agent_response": "DM us tracking number.", "intent": "delivery_issue"},
                {"conversation_id": "d2", "customer_message": "I was charged twice", "agent_response": "Refund issued.", "intent": "refund_request"},
                {"conversation_id": "d3", "customer_message": "Cancel my subscription", "agent_response": "Cancelled.", "intent": "cancellation"},
                {"conversation_id": "d4", "customer_message": "Unauthorized charge!", "agent_response": "DM us immediately.", "intent": "account_security"}
            ]

        self.majority.fit(
            training_intents=[it["intent"] for it in dev_items],
            training_responses=[it["agent_response"] for it in dev_items]
        )
        self.tfidf.fit(dev_items)

        results = {}
        systems = {
            "Majority": self.majority,
            "TF-IDF": self.tfidf,
            "Embedding Retrieval": self.emb_retrieval,
            "AI Agent": self.agent
        }

        for sys_name, sys_obj in systems.items():
            print(f"Evaluating System: {sys_name} on {len(records)} golden records...")
            gold_intents = []
            pred_intents = []
            gold_decisions = []
            pred_decisions = []
            eval_records = []
            judge_scores = []

            for r in records:
                msg = r["message"]
                g_intent = r["gold_intent"]
                g_dec = r["gold_escalation"]
                exclude_convs = [r.get("conversation_id", "")]

                # Process system
                if sys_name in ["Majority", "TF-IDF", "Embedding Retrieval"]:
                    if hasattr(sys_obj, "process"):
                        try:
                            resp = sys_obj.process(msg, exclude_conv_ids=exclude_convs)
                        except TypeError:
                            resp = sys_obj.process(msg)
                else:
                    resp = self.agent.process(msg, exclude_conv_ids=exclude_convs)

                p_intent = resp.intent.get("name", "unknown_other")
                p_dec = resp.decision

                gold_intents.append(g_intent)
                pred_intents.append(p_intent)
                gold_decisions.append(g_dec)
                pred_decisions.append(p_dec)

                eval_records.append({
                    "id": r.get("id", ""),
                    "message": msg,
                    "gold_intent": g_intent,
                    "pred_intent": p_intent,
                    "gold_escalation": g_dec,
                    "pred_escalation": p_dec,
                    "system_reply": resp.reply,
                    "decision_reason": resp.decision_reason
                })

                # Run LLM judge sample on subset
                if sys_name == "AI Agent" and len(judge_scores) < 30:
                    ev_res = self.judge.evaluate_response(
                        customer_message=msg,
                        expected_intent=g_intent,
                        system_intent=p_intent,
                        historical_evidence=r.get("reference_resolution", ""),
                        system_reply=resp.reply,
                        system_decision=p_dec,
                        gold_decision=g_dec
                    )
                    judge_scores.append(ev_res.overall)

            intent_m = EvaluationMetricsCalculator.calculate_intent_metrics(gold_intents, pred_intents)
            esc_m = EvaluationMetricsCalculator.calculate_escalation_metrics(gold_decisions, pred_decisions)
            avg_reply_score = round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None

            results[sys_name] = {
                "intent_metrics": intent_m,
                "escalation_metrics": esc_m,
                "avg_reply_score": avg_reply_score,
                "records": eval_records
            }

        return results

    @staticmethod
    def _infer_dev_intent(message: str) -> str:
        """Infer intent from dev conversation message text using keyword heuristics.
        
        This ensures TF-IDF baseline trains on diverse intents instead of
        a single uniform class which would prevent classifier fitting.
        """
        msg = message.lower()
        if any(kw in msg for kw in ["unauthorized", "stolen", "fraud", "hacked"]):
            return "account_security"
        if any(kw in msg for kw in ["refund", "charged", "billed", "reimburse"]):
            return "refund_request"
        if any(kw in msg for kw in ["cancel", "cancellation", "unsubscribe"]):
            return "cancellation"
        if any(kw in msg for kw in ["broken", "damaged", "shattered", "cracked", "defective"]):
            return "damaged_item"
        if any(kw in msg for kw in ["wrong item", "wrong color", "wrong size", "wrong product"]):
            return "wrong_item_received"
        if any(kw in msg for kw in ["delivery", "package", "shipment", "tracking", "arrived", "shipped"]):
            return "delivery_issue"
        if any(kw in msg for kw in ["password", "login", "log in", "cannot access", "account locked"]):
            return "account_access"
        if any(kw in msg for kw in ["price", "cost", "how much", "subscription fee"]):
            return "pricing_question"
        if any(kw in msg for kw in ["crash", "error", "bug", "not working", "buffering"]):
            return "technical_problem"
        if any(kw in msg for kw in ["worst", "terrible", "awful", "rude", "horrible"]):
            return "complaint"
        if any(kw in msg for kw in ["thank", "great", "amazing", "helpful", "awesome"]):
            return "feedback_praise"
        return "unknown_other"

