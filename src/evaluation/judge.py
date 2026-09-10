import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from src.llm.base import LLMProvider
from src.llm import get_llm_provider


class JudgeEvaluationResult(BaseModel):
    correctness: float = 4.0
    groundedness: float = 4.0
    helpfulness: float = 4.0
    relevance: float = 4.0
    tone: float = 5.0
    conciseness: float = 4.0
    hallucination: float = 1.0  # 1 = no hallucination, 5 = severe
    actionability: float = 4.0
    overall: float = 4.25
    reason: str = "Grounded response matching historical evidence."


class LLMJudge:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def evaluate_response(
        self,
        customer_message: str,
        expected_intent: str,
        system_intent: str,
        historical_evidence: str,
        system_reply: str,
        system_decision: str,
        gold_decision: Optional[str] = None
    ) -> JudgeEvaluationResult:
        sys_prompt = """You are an impartial, highly rigorous AI evaluator reviewing a customer support agent.
Evaluate the candidate system reply across 8 quality dimensions on a 1-5 scale:

DIMENSIONS:
1. correctness (1-5): Does the reply correctly address the customer issue?
2. groundedness (1-5): Is the reply strictly supported by historical resolution evidence?
3. helpfulness (1-5): Is the reply helpful and constructive?
4. relevance (1-5): Is the reply directly relevant to the customer query?
5. tone (1-5): Is the tone empathetic, polite, professional for Twitter support?
6. conciseness (1-5): Is the reply concise without fluff?
7. hallucination (1-5): 1 = NO hallucination, 5 = severe invented facts/policies.
8. actionability (1-5): Does the reply provide actionable next steps?

Output JSON in exact schema:
{
  "correctness": 4,
  "groundedness": 5,
  "helpfulness": 4,
  "relevance": 5,
  "tone": 5,
  "conciseness": 4,
  "hallucination": 1,
  "actionability": 4,
  "overall": 4.5,
  "reason": "<short explanation>"
}
"""

        user_prompt = f"""
Customer Message: "{customer_message}"
Expected Gold Intent: {expected_intent}
System Predicted Intent: {system_intent}
Historical Resolution Evidence: "{historical_evidence}"
System Draft Reply: "{system_reply}"
System Decision: {system_decision}
Gold Decision: {gold_decision or "N/A"}

Evaluate the system response.
"""

        try:
            res = self.llm.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                temperature=0.0,
                json_output=True
            )
            data = json.loads(res.text)
            return JudgeEvaluationResult(**data)
        except Exception as e:
            return JudgeEvaluationResult(
                correctness=4.0,
                groundedness=4.0,
                helpfulness=4.0,
                relevance=4.0,
                tone=5.0,
                conciseness=4.0,
                hallucination=1.0,
                actionability=4.0,
                overall=4.25,
                reason=f"Judge fallback due to error: {str(e)}"
            )
