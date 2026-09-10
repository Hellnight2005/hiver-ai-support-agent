from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GroundingEvidence(BaseModel):
    conversation_id: str
    similarity: float
    customer_message: str
    agent_response: str


class EscalationDecision(BaseModel):
    decision: str  # "AUTO_HANDLE" | "ESCALATE"
    reason: str
    signals: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    message: str
    intent: Dict[str, Any]
    retrieval: List[GroundingEvidence] = Field(default_factory=list)
    reply: str
    decision: str  # "AUTO_HANDLE" | "ESCALATE"
    decision_reason: str
    risk_flags: List[str] = Field(default_factory=list)
