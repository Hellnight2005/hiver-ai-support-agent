import pytest
from src.data.loader import DataLoader
from src.data.cleaner import TextCleaner
from src.data.conversations import ConversationReconstructor, analyze_brand_suitability
from src.data.schemas import Tweet


def test_text_cleaner():
    raw = "  Hello   world \0 null char!  "
    cleaned = TextCleaner.clean_text(raw)
    assert cleaned == "Hello world null char!"


def test_tweet_loader_with_sample():
    loader = DataLoader()
    tweets = loader.load_tweets()
    assert len(tweets) > 0
    assert isinstance(tweets[0], Tweet)


def test_conversation_reconstruction():
    t1 = Tweet(tweet_id="1", author_id="cust1", inbound=True, created_at="2017", text="Help me", in_response_to_tweet_id=None)
    t2 = Tweet(tweet_id="2", author_id="AmazonHelp", inbound=False, created_at="2017", text="DM us", in_response_to_tweet_id="1", brand="AmazonHelp")
    
    reconstructor = ConversationReconstructor([t1, t2])
    convs = reconstructor.reconstruct_all()

    assert len(convs) == 1
    c = convs[0]
    assert c.brand == "AmazonHelp"
    assert c.turn_count == 2
    assert c.is_multi_turn is True
    assert c.initial_customer_message == "Help me"
    assert c.first_agent_response == "DM us"


def test_brand_suitability_analysis():
    t1 = Tweet(tweet_id="1", author_id="cust1", inbound=True, created_at="2017", text="Help me", in_response_to_tweet_id=None)
    t2 = Tweet(tweet_id="2", author_id="AmazonHelp", inbound=False, created_at="2017", text="DM us", in_response_to_tweet_id="1", brand="AmazonHelp")
    
    reconstructor = ConversationReconstructor([t1, t2])
    convs = reconstructor.reconstruct_all()

    stats = analyze_brand_suitability(convs)
    assert len(stats) == 1
    assert stats[0].brand_name == "AmazonHelp"
    assert stats[0].total_conversations == 1
