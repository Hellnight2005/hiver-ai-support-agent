import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.intents.taxonomy import IntentTaxonomy
from src.llm.base import LLMProvider


class AlternativeIntent(BaseModel):
    intent: str
    confidence: float


class IntentClassificationResult(BaseModel):
    intent: str
    confidence: float
    reason: str
    alternative_intents: List[AlternativeIntent] = Field(default_factory=list)


class LLMIntentClassifier:
    def __init__(self, llm_provider: LLMProvider, taxonomy: Optional[IntentTaxonomy] = None):
        self.llm = llm_provider
        self.taxonomy = taxonomy or IntentTaxonomy.load_from_file()

    def classify(self, customer_message: str) -> IntentClassificationResult:
        system_prompt = self._build_system_prompt()
        user_prompt = f"Customer Message: \"{customer_message}\"\n\nClassify this message."

        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                json_output=True
            )
            data = json.loads(response.text)
            
            intent_name = data.get("intent", "unknown_other")
            if not self.taxonomy.validate_intent(intent_name):
                intent_name = "unknown_other"

            confidence = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", "No reason provided."))

            alts = []
            for alt in data.get("alternative_intents", []):
                alts.append(AlternativeIntent(
                    intent=str(alt.get("intent", "unknown_other")),
                    confidence=float(alt.get("confidence", 0.0))
                ))

            return IntentClassificationResult(
                intent=intent_name,
                confidence=confidence,
                reason=reason,
                alternative_intents=alts
            )
        except Exception as e:
            return IntentClassificationResult(
                intent="unknown_other",
                confidence=0.0,
                reason=f"Classification fallback due to error: {str(e)}",
                alternative_intents=[]
            )

    def _build_system_prompt(self) -> str:
        intent_lines = []
        for i in self.taxonomy.intents:
            intent_lines.append(f"- {i.name}: {i.definition} Examples: {', '.join(i.examples[:2])}")

        taxonomy_str = "\n".join(intent_lines)

        return f"""You are an expert customer support intent classifier.
Classify the customer message into EXACTLY ONE primary intent from the allowed taxonomy below.

ALLOWED INTENT TAXONOMY:
{taxonomy_str}

RULES:
1. Choose exactly one primary intent from the allowed list.
2. If evidence is insufficient, ambiguous, or gibberish, select "unknown_other".
3. Confidence must be between 0.0 and 1.0 based on clarity.
4. Output MUST be valid JSON in the following format:

{{
  "intent": "<intent_name>",
  "confidence": 0.85,
  "reason": "<short diagnostic explanation>",
  "alternative_intents": [
    {{"intent": "<alt_intent>", "confidence": 0.10}}
  ]
}}
"""
