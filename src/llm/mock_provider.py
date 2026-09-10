import json
import re
from typing import Optional
from src.llm.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    def __init__(self, model: str = "mock-model"):
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 300,
        json_output: bool = False
    ) -> LLMResponse:
        prompt_lower = prompt.lower()
        sys_lower = (system_prompt or "").lower()

        # Intent classification request
        if "classify" in prompt_lower or "intent" in sys_lower or "taxonomy" in prompt_lower:
            text = self._mock_intent_response(prompt_lower)
        # LLM Judge request
        elif "judge" in sys_lower or "evaluate the following" in prompt_lower or "correctness" in prompt_lower:
            text = self._mock_judge_response(prompt_lower)
        # Response generation request
        elif "draft" in prompt_lower or "agent" in sys_lower or "customer support" in sys_lower:
            text = self._mock_generation_response(prompt_lower, json_output)
        else:
            if json_output:
                text = json.dumps({"status": "ok", "message": "Mock default response."})
            else:
                text = "Mock default response."

        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split())
        )

    def _mock_intent_response(self, prompt_lower: str) -> str:
        # Default to valid taxonomy intent (not non-existent 'information_request')
        intent = "unknown_other"
        confidence = 0.55
        reason = "No strong keyword signal detected; defaulting to unknown."

        # Order matters: check high-risk/specific intents first, then broader ones
        if "unauthorized" in prompt_lower or "stolen" in prompt_lower or "fraud" in prompt_lower or "hacked" in prompt_lower:
            intent = "account_security"
            confidence = 0.95
            reason = "High risk security or fraud issue detected."
        elif "refund" in prompt_lower or "charged" in prompt_lower or "billed" in prompt_lower or "reimburse" in prompt_lower:
            intent = "refund_request"
            confidence = 0.92
            reason = "Customer explicitly mentions refund or billing charge."
        elif "wrong item" in prompt_lower or "wrong color" in prompt_lower or "wrong size" in prompt_lower or "wrong product" in prompt_lower:
            intent = "wrong_item_received"
            confidence = 0.91
            reason = "Customer received an item different from their order."
        elif "broken" in prompt_lower or "damaged" in prompt_lower or "defective" in prompt_lower or "shattered" in prompt_lower or "cracked" in prompt_lower:
            intent = "damaged_item"
            confidence = 0.94
            reason = "Customer reports receiving damaged or broken merchandise."
        elif "delivery" in prompt_lower or "arrived" in prompt_lower or "shipment" in prompt_lower or "tracking" in prompt_lower or "package" in prompt_lower or "shipped" in prompt_lower:
            intent = "delivery_issue"
            confidence = 0.89
            reason = "Customer asks about delayed or missing package delivery."
        elif "cancel" in prompt_lower or "unsubscribe" in prompt_lower:
            intent = "cancellation"
            confidence = 0.91
            reason = "Customer requests order or subscription cancellation."
        elif "password" in prompt_lower or "login" in prompt_lower or "log in" in prompt_lower or "cannot access" in prompt_lower or "locked" in prompt_lower:
            intent = "account_access"
            confidence = 0.90
            reason = "Customer reports login or account access difficulties."
        elif "price" in prompt_lower or "cost" in prompt_lower or "how much" in prompt_lower or "subscription fee" in prompt_lower or "pricing" in prompt_lower:
            intent = "pricing_question"
            confidence = 0.88
            reason = "Customer asks about product or service pricing."
        elif "crash" in prompt_lower or "error" in prompt_lower or "bug" in prompt_lower or "not working" in prompt_lower or "buffering" in prompt_lower or "glitch" in prompt_lower or "pause" in prompt_lower or "pausing" in prompt_lower:
            intent = "technical_problem"
            confidence = 0.87
            reason = "Customer reports app crash, error, or technical malfunction."
        elif "worst" in prompt_lower or "terrible" in prompt_lower or "awful" in prompt_lower or "rude" in prompt_lower or "horrible" in prompt_lower or "hate" in prompt_lower:
            intent = "complaint"
            confidence = 0.86
            reason = "Customer expresses explicit dissatisfaction with service."
        elif "thank" in prompt_lower or "great job" in prompt_lower or "amazing" in prompt_lower or "helpful" in prompt_lower or "awesome" in prompt_lower:
            intent = "feedback_praise"
            confidence = 0.85
            reason = "Customer provides positive feedback or appreciation."
        elif "random gibberish" in prompt_lower or "asdf" in prompt_lower:
            intent = "unknown_other"
            confidence = 0.40
            reason = "Insufficient evidence to map to standard intent."

        return json.dumps({
            "intent": intent,
            "confidence": confidence,
            "reason": reason,
            "alternative_intents": [
                {"intent": "unknown_other", "confidence": 0.08}
            ]
        })

    def _mock_generation_response(self, prompt_lower: str, json_output: bool) -> str:
        # Extract predicted intent from user prompt if present
        intent = ""
        intent_match = re.search(r"predicted intent:\s*([a-z_]+)", prompt_lower)
        if intent_match:
            intent = intent_match.group(1).strip()

        # Extract customer message
        cust_msg = ""
        msg_match = re.search(r'customer message:\s*"([^"]+)"', prompt_lower)
        if msg_match:
            cust_msg = msg_match.group(1).strip()

        if intent == "account_security" or any(kw in cust_msg for kw in ["unauthorized", "stolen", "fraud", "hacked", "identity theft"]):
            reply = "We take account security very seriously. Please immediately secure your account, change your password, and DM us so our security team can assist you."
        elif intent == "refund_request" or any(kw in cust_msg for kw in ["refund", "charged", "billed", "reimburse", "duplicate"]):
            reply = "We apologize for the billing concern. Please DM us your account email and order details so we can verify the transaction and process any applicable refund."
        elif intent == "delivery_issue" or any(kw in cust_msg for kw in ["delivery", "package", "shipment", "arrived", "tracking"]):
            reply = "We apologize for the delivery delay. Please send us a DM with your tracking number so we can update your shipment status."
        elif intent == "technical_problem" or any(kw in cust_msg for kw in ["crash", "pause", "bug", "error", "glitch", "app", "ios", "android"]):
            reply = "We are sorry for the technical trouble. Please try restarting your app/device, or DM us your device model and OS version so our technical support team can troubleshoot."
        elif intent == "complaint" or any(kw in cust_msg for kw in ["worst", "terrible", "horrible", "unhelpful", "rude", "awful"]):
            reply = "We sincerely apologize for your poor experience. Your concern has been prioritized, and a senior human support specialist will assist you via DM."
        elif intent == "cancellation" or any(kw in cust_msg for kw in ["cancel", "cancellation", "unsubscribe"]):
            reply = "You can cancel unshipped orders in Your Orders, or send us a direct message with your order number so we can assist you with cancellation."
        elif intent == "damaged_item" or any(kw in cust_msg for kw in ["broken", "damaged", "shattered", "defective"]):
            reply = "We are very sorry to hear your item arrived damaged. Please send us a DM with your order ID and a photo so we can arrange a free replacement."
        elif intent == "wrong_item_received" or "wrong item" in cust_msg:
            reply = "We apologize for the mix up! Please send us a DM with your order details so we can ship the correct item to you right away."
        elif intent == "pricing_question" or any(kw in cust_msg for kw in ["price", "cost", "how much", "fee"]):
            reply = "For current pricing, tier plans, and promotional discounts, please visit our official pricing page or DM us with your query."
        elif intent == "feedback_praise" or any(kw in cust_msg for kw in ["thank", "great", "awesome", "helpful", "amazing"]):
            reply = "Thank you so much for the kind feedback! We are always delighted to help. Have a wonderful day!"
        else:
            reply = "Thank you for contacting customer support. Please send us a direct message with your order details so our team can assist you."

        if json_output:
            return json.dumps({
                "reply": reply,
                "grounding_evidence": [{"conversation_id": "conv_mock_1", "similarity": 0.85}],
                "confidence": 0.88
            })
        return reply

    def _mock_judge_response(self, prompt_lower: str) -> str:
        return json.dumps({
            "correctness": 4,
            "groundedness": 5,
            "helpfulness": 4,
            "relevance": 5,
            "tone": 5,
            "conciseness": 4,
            "hallucination": 1,
            "actionability": 4,
            "overall": 4.4,
            "reason": "The system reply addresses the customer message politely and matches historical resolution guidance without introducing unsupported claims."
        })
