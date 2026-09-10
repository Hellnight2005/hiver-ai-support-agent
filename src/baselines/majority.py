from typing import List, Dict, Any, Optional
from collections import Counter
from src.agent.schemas import AgentResponse, GroundingEvidence


class MajorityBaseline:
    def __init__(self):
        self.most_common_intent = "delivery_issue"
        self.default_reply = "We apologize for the trouble! Please DM us your order details so we can assist you."

    def fit(self, training_intents: List[str], training_responses: List[str]):
        if training_intents:
            intent_counts = Counter(training_intents)
            self.most_common_intent = intent_counts.most_common(1)[0][0]
        if training_responses:
            resp_counts = Counter(training_responses)
            self.default_reply = resp_counts.most_common(1)[0][0]

    def process(self, customer_message: str) -> AgentResponse:
        return AgentResponse(
            message=customer_message,
            intent={
                "name": self.most_common_intent,
                "confidence": 0.50,
                "reason": "Majority baseline rule (always predicts most common class)."
            },
            retrieval=[],
            reply=self.default_reply,
            decision="AUTO_HANDLE",
            decision_reason="Majority baseline default auto-handle.",
            risk_flags=[]
        )
