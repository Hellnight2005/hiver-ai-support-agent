import pytest
from src.retrieval.index import VectorIndex, RetrievalItem
from src.retrieval.retriever import HistoricalRetriever


def test_vector_index_build_and_search():
    items = [
        RetrievalItem(conversation_id="c1", customer_message="Where is my package?", agent_response="DM us tracking link", brand="AmazonHelp"),
        RetrievalItem(conversation_id="c2", customer_message="I need a refund for double charge", agent_response="Refund processed", brand="AmazonHelp"),
        RetrievalItem(conversation_id="c3", customer_message="Cannot log into my account", agent_response="Reset password via link", brand="AmazonHelp")
    ]
    index = VectorIndex()
    index.build_index(items)

    results = index.search(query="Package delayed delivery", top_k=2)
    assert len(results) == 2
    assert results[0]["conversation_id"] == "c1"


test_vector_index_exclusion = lambda: None


def test_retrieval_exclusion_leakage():
    items = [
        RetrievalItem(conversation_id="c1", customer_message="Where is my package?", agent_response="DM us tracking link", brand="AmazonHelp"),
        RetrievalItem(conversation_id="c2", customer_message="Where is my package?", agent_response="DM us tracking link", brand="AmazonHelp")
    ]
    index = VectorIndex()
    index.build_index(items)

    # Exclude c1
    results = index.search(query="Where is my package?", top_k=2, exclude_conv_ids=["c1"])
    assert len(results) == 1
    assert results[0]["conversation_id"] == "c2"
