from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from src.agent.agent import SupportAgent
from src.intents.taxonomy import IntentTaxonomy
from src.config import settings

app = FastAPI(
    title="Hiver AI Support Agent API",
    description="Grounded AI Support Agent with Intent Classification, Historical Retrieval, and Escalation Policy",
    version="1.0.0"
)

agent_instance = None


def get_agent() -> SupportAgent:
    global agent_instance
    if agent_instance is None:
        agent_instance = SupportAgent()
    return agent_instance


class SupportRequest(BaseModel):
    message: str = Field(..., min_length=1, json_schema_extra={"example": "I have been charged twice for the same transaction"})


class SupportResponse(BaseModel):
    intent: Dict[str, Any]
    reply: str
    decision: str
    decision_reason: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "brand": settings.brand.default_brand,
        "model": settings.llm.model,
        "embedding_model": settings.retrieval.embedding_model
    }


@app.get("/v1/intents", status_code=status.HTTP_200_OK)
def list_intents():
    taxonomy = IntentTaxonomy.load_from_file()
    return {
        "count": len(taxonomy.intents),
        "intents": [
            {
                "name": i.name,
                "definition": i.definition,
                "examples": i.examples
            }
            for i in taxonomy.intents
        ]
    }


@app.post("/v1/support", response_model=SupportResponse, status_code=status.HTTP_200_OK)
def process_support_query(request: SupportRequest):
    agent = get_agent()
    try:
        res = agent.process(request.message)
        evidence_list = [
            {
                "conversation_id": ev.conversation_id,
                "similarity": round(ev.similarity, 4),
                "customer_message": ev.customer_message,
                "agent_response": ev.agent_response
            }
            for ev in res.retrieval
        ]

        return SupportResponse(
            intent=res.intent,
            reply=res.reply,
            decision=res.decision,
            decision_reason=res.decision_reason,
            evidence=evidence_list,
            risk_flags=res.risk_flags
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing support query: {str(e)}"
        )


@app.get("/v1/evaluation", status_code=status.HTTP_200_OK)
def get_evaluation_metrics():
    """Retrieve the latest dynamic evaluation results artifact."""
    import json
    from pathlib import Path
    eval_file = Path(settings.paths.evaluation_results)
    if not eval_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation results artifact not found. Please run scripts/run_evaluation.py first."
        )
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def get_dashboard():
    """Serve the visual interactive dashboard."""
    from fastapi.responses import FileResponse
    from pathlib import Path
    dashboard_file = Path("dashboard/index.html")
    if dashboard_file.exists():
        return FileResponse(dashboard_file)
    return {"message": "Hiver Support Agent API is running. Visit /docs for OpenAPI specifications."}
