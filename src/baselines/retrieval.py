from typing import List, Optional
from src.retrieval.retriever import HistoricalRetriever
from src.agent.schemas import AgentResponse, GroundingEvidence


class EmbeddingRetrievalBaseline:
    def __init__(self, retriever: Optional[HistoricalRetriever] = None):
        self.retriever = retriever or HistoricalRetriever()

    def process(self, customer_message: str, exclude_conv_ids: Optional[List[str]] = None) -> AgentResponse:
        retrieved = self.retriever.retrieve(
            customer_message=customer_message,
            top_k=5,
            exclude_conv_ids=exclude_conv_ids
        )

        evidence = [
            GroundingEvidence(
                conversation_id=r["conversation_id"],
                similarity=r["similarity"],
                customer_message=r["customer_message"],
                agent_response=r["agent_response"]
            )
            for r in retrieved
        ]

        if evidence:
            top_ev = evidence[0]
            top_sim = top_ev.similarity
            reply = top_ev.agent_response
            decision = "AUTO_HANDLE" if top_sim >= 0.75 else "ESCALATE"
            intent_name = retrieved[0].get("intent", "unknown_other")
        else:
            top_sim = 0.0
            reply = "Please DM us your details."
            decision = "ESCALATE"
            intent_name = "unknown_other"

        return AgentResponse(
            message=customer_message,
            intent={"name": intent_name, "confidence": top_sim, "reason": "Embedding retrieval nearest match"},
            retrieval=evidence,
            reply=reply,
            decision=decision,
            decision_reason=f"Embedding retrieval score = {top_sim:.2f}",
            risk_flags=[]
        )
