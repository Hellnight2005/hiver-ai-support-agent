import pytest
from src.intents.taxonomy import IntentTaxonomy
from src.intents.classifier import LLMIntentClassifier
from src.llm.mock_provider import MockProvider


def test_intent_taxonomy_schema():
    taxonomy = IntentTaxonomy.load_from_file()
    names = taxonomy.get_intent_names()
    assert "delivery_issue" in names
    assert "unknown_other" in names
    assert len(names) >= 8


def test_intent_classifier_with_mock():
    llm = MockProvider()
    classifier = LLMIntentClassifier(llm_provider=llm)

    res = classifier.classify("I was charged twice for my order, refund please!")
    assert res.intent == "refund_request"
    assert res.confidence >= 0.80
    assert "refund" in res.reason.lower() or "billing" in res.reason.lower()


def test_intent_classifier_unknown():
    llm = MockProvider()
    classifier = LLMIntentClassifier(llm_provider=llm)

    res = classifier.classify("asdfghjkl random unknown test string")
    assert res.intent == "unknown_other"
