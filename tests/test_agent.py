import pytest
from src.agent.agent import SupportAgent
from src.agent.schemas import AgentResponse
from src.llm.mock_provider import MockProvider


def test_agent_end_to_end_mock():
    agent = SupportAgent(llm_provider=MockProvider())
    res = agent.process("I was charged twice for my subscription, refund please")

    assert isinstance(res, AgentResponse)
    assert res.intent["name"] == "refund_request"
    assert res.decision in ["AUTO_HANDLE", "ESCALATE"]
    assert len(res.reply) > 0
    assert res.decision_reason != ""
