import pytest
from src.agent.policy import EscalationPolicyEngine
from src.agent.schemas import GroundingEvidence


def test_policy_high_confidence_auto_handle():
    policy = EscalationPolicyEngine(intent_threshold=0.80, retrieval_threshold=0.75)
    evidence = [GroundingEvidence(conversation_id="c1", similarity=0.85, customer_message="msg", agent_response="resp")]
    
    decision = policy.evaluate(
        customer_message="My package hasn't arrived",
        predicted_intent="delivery_issue",
        intent_confidence=0.90,
        retrieval_evidence=evidence
    )
    assert decision.decision == "AUTO_HANDLE"


def test_policy_low_intent_confidence_escalate():
    policy = EscalationPolicyEngine(intent_threshold=0.80, retrieval_threshold=0.75)
    evidence = [GroundingEvidence(conversation_id="c1", similarity=0.85, customer_message="msg", agent_response="resp")]
    
    decision = policy.evaluate(
        customer_message="Some vague msg",
        predicted_intent="delivery_issue",
        intent_confidence=0.60,
        retrieval_evidence=evidence
    )
    assert decision.decision == "ESCALATE"
    assert "confidence" in decision.reason.lower()


def test_policy_unknown_intent_escalate():
    policy = EscalationPolicyEngine()
    evidence = [GroundingEvidence(conversation_id="c1", similarity=0.90, customer_message="msg", agent_response="resp")]
    
    decision = policy.evaluate(
        customer_message="asdfghjkl",
        predicted_intent="unknown_other",
        intent_confidence=0.90,
        retrieval_evidence=evidence
    )
    assert decision.decision == "ESCALATE"
    assert "unknown_other" in decision.reason.lower()


def test_policy_sensitive_category_escalate():
    policy = EscalationPolicyEngine(sensitive_intents=["account_security"])
    evidence = [GroundingEvidence(conversation_id="c1", similarity=0.90, customer_message="msg", agent_response="resp")]
    
    decision = policy.evaluate(
        customer_message="Someone stole my account",
        predicted_intent="account_security",
        intent_confidence=0.95,
        retrieval_evidence=evidence
    )
    assert decision.decision == "ESCALATE"
    assert "sensitive" in decision.reason.lower()


def test_policy_risk_keyword_escalate():
    policy = EscalationPolicyEngine()
    evidence = [GroundingEvidence(conversation_id="c1", similarity=0.90, customer_message="msg", agent_response="resp")]
    
    decision = policy.evaluate(
        customer_message="I am going to sue you with my attorney!",
        predicted_intent="delivery_issue",
        intent_confidence=0.95,
        retrieval_evidence=evidence
    )
    assert decision.decision == "ESCALATE"
    assert "attorney" in str(decision.signals.get("risk_flags", []))
