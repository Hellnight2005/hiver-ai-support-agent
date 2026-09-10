import json
from typing import List, Optional, Dict, Any
from src.config import settings
from src.llm.base import LLMProvider
from src.llm import get_llm_provider
from src.data.cleaner import TextCleaner
from src.intents.classifier import LLMIntentClassifier
from src.retrieval.retriever import HistoricalRetriever
from src.retrieval.index import VectorIndex
from src.agent.policy import EscalationPolicyEngine
from src.agent.prompting import PromptBuilder
from src.agent.schemas import AgentResponse, GroundingEvidence, EscalationDecision


class SupportAgent:
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        retriever: Optional[HistoricalRetriever] = None,
        policy_engine: Optional[EscalationPolicyEngine] = None,
        brand_name: Optional[str] = None
    ):
        self.llm = llm_provider or get_llm_provider()
        self.classifier = LLMIntentClassifier(llm_provider=self.llm)
        self.retriever = retriever or HistoricalRetriever()
        
        self.policy = policy_engine or EscalationPolicyEngine(
            intent_threshold=settings.policy.auto_handle_intent_threshold,
            retrieval_threshold=settings.policy.auto_handle_retrieval_threshold,
            sensitive_intents=settings.policy.sensitive_intents
        )
        self.brand_name = brand_name or settings.brand.default_brand

    def process(
        self,
        customer_message: str,
        exclude_conv_ids: Optional[List[str]] = None
    ) -> AgentResponse:
        # Step 1: Preprocess message
        clean_msg = TextCleaner.clean_text(customer_message)

        # Step 2: Intent Classification
        classification = self.classifier.classify(clean_msg)

        # Step 3: Historical Retrieval
        retrieved_raw = self.retriever.retrieve(
            customer_message=clean_msg,
            top_k=settings.retrieval.top_k,
            exclude_conv_ids=exclude_conv_ids
        )

        evidence = [
            GroundingEvidence(
                conversation_id=r["conversation_id"],
                similarity=r["similarity"],
                customer_message=r["customer_message"],
                agent_response=r["agent_response"]
            )
            for r in retrieved_raw
        ]

        # Step 4: Escalation Policy Decision
        policy_decision = self.policy.evaluate(
            customer_message=clean_msg,
            predicted_intent=classification.intent,
            intent_confidence=classification.confidence,
            retrieval_evidence=evidence
        )

        # Step 5: Draft Response Generation
        draft_reply = self._generate_response(
            customer_message=clean_msg,
            intent=classification.intent,
            evidence=evidence,
            decision=policy_decision.decision
        )

        # Build final response payload
        return AgentResponse(
            message=clean_msg,
            intent={
                "name": classification.intent,
                "confidence": classification.confidence,
                "reason": classification.reason
            },
            retrieval=evidence,
            reply=draft_reply,
            decision=policy_decision.decision,
            decision_reason=policy_decision.reason,
            risk_flags=policy_decision.signals.get("risk_flags", [])
        )

    def _generate_response(
        self,
        customer_message: str,
        intent: str,
        evidence: List[GroundingEvidence],
        decision: str
    ) -> str:
        sys_prompt = PromptBuilder.build_generation_system_prompt(self.brand_name)
        user_prompt = PromptBuilder.build_generation_user_prompt(customer_message, intent, evidence)
        if decision == "ESCALATE":
            user_prompt += "\nNote: This issue has been routed for human specialist review. Draft a polite, empathetic response acknowledging their specific issue and inviting them to DM their details for human follow-up."

        try:
            res = self.llm.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                temperature=0.0,
                json_output=True
            )
            data = json.loads(res.text)
            return data.get("reply", "Thank you for reaching out. Please DM us your order details so we can assist you.")
        except Exception:
            # Fallback to intent-specific empathetic response
            if intent == "delivery_issue":
                return "We apologize for the delivery delay. Please send us a direct message with your tracking and order ID so we can investigate and assist you right away."
            if intent == "refund_request":
                return "We apologize for the billing concern. Please DM us your account email so we can verify the transaction and process any applicable refund."
            if intent == "account_security":
                return "We take account security very seriously. Please immediately secure your account, change your password, and DM us so our security team can assist you."
            if intent == "technical_problem":
                return "We are sorry for the technical trouble. Please try restarting your app/device, or DM us your device model and OS version so our technical support team can troubleshoot."
            if intent == "complaint":
                return "We sincerely apologize for your poor experience. Your concern has been prioritized, and a senior human support specialist will assist you via DM."
            if intent == "cancellation":
                return "You can cancel unshipped orders in Your Orders, or send us a direct message with your order number so we can assist you with cancellation."
            if intent == "damaged_item":
                return "We are very sorry to hear your item arrived damaged. Please send us a DM with your order ID and a photo so we can arrange a free replacement."
            if intent == "wrong_item_received":
                return "We apologize for the mix up! Please send us a DM with your order details so we can ship the correct item to you right away."
            if intent == "pricing_question":
                return "For current pricing, tier plans, and promotional discounts, please visit our official pricing page or DM us with your query."
            if intent == "feedback_praise":
                return "Thank you so much for the kind feedback! We are always delighted to help. Have a wonderful day!"
            if intent == "account_access":
                return "We can help you regain access to your account. Please click 'Forgot Password' on the login screen or send us a DM with your registered email."
            if evidence:
                return f"Apologies for the trouble. {evidence[0].agent_response}"
            return "Thank you for reaching out to customer support. Could you please provide a few more details or DM us your question so we can assist you?"
