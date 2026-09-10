import re
from typing import List, Dict, Any
from src.agent.schemas import EscalationDecision, GroundingEvidence
from src.config import settings


class EscalationPolicyEngine:
    def __init__(
        self,
        intent_threshold: float = 0.80,
        retrieval_threshold: float = 0.20,
        sensitive_intents: List[str] = None
    ):
        self.intent_threshold = intent_threshold
        self.retrieval_threshold = retrieval_threshold
        self.sensitive_intents = sensitive_intents or [
            "account_security",
            "complaint",
            "fraud_report",
            "legal_complaint"
        ]
        self.risk_keywords = [
            "lawyer", "attorney", "sue", "legal action", "police", "court",
            "fraud", "stolen", "hacked", "identity theft"
        ]

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_evidence: List[GroundingEvidence]
    ) -> EscalationDecision:
        risk_flags = []
        msg_lower = customer_message.lower()

        # Check risk keywords
        for kw in self.risk_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", msg_lower):
                risk_flags.append(f"risk_keyword:{kw}")

        top_similarity = retrieval_evidence[0].similarity if retrieval_evidence else 0.0
        evidence_count = len(retrieval_evidence)

        signals = {
            "predicted_intent": predicted_intent,
            "intent_confidence": round(intent_confidence, 2),
            "top_similarity": round(top_similarity, 2),
            "evidence_count": evidence_count,
            "risk_flags": risk_flags
        }

        # Rule 1: High risk keywords
        if risk_flags:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"High-risk keyword detected ({', '.join(risk_flags)}). Requires human agent review.",
                signals=signals
            )

        # Rule 2: Sensitive intent category
        if predicted_intent.lower() in [i.lower() for i in self.sensitive_intents]:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Intent '{predicted_intent}' is classified as a sensitive category requiring manual support.",
                signals=signals
            )

        # Rule 3: Unknown intent
        if predicted_intent.lower() == "unknown_other":
            return EscalationDecision(
                decision="ESCALATE",
                reason="Customer intent could not be determined with high certainty (unknown_other).",
                signals=signals
            )

        # Rule 4: Low intent confidence
        if intent_confidence < self.intent_threshold:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Intent confidence ({intent_confidence:.2f}) is below auto-handling threshold ({self.intent_threshold:.2f}).",
                signals=signals
            )

        # Rule 5: Insufficient retrieval evidence
        if evidence_count < 1 or top_similarity < self.retrieval_threshold:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Historical resolution similarity ({top_similarity:.2f}) is below retrieval threshold ({self.retrieval_threshold:.2f}).",
                signals=signals
            )

        # Default: AUTO_HANDLE
        return EscalationDecision(
            decision="AUTO_HANDLE",
            reason="High intent confidence and strong historical resolution evidence met all automation policy criteria.",
            signals=signals
        )
