from typing import List, Optional
from pydantic import BaseModel, Field


class Tweet(BaseModel):
    tweet_id: str
    author_id: str
    inbound: bool
    created_at: str
    text: str
    response_tweet_id: Optional[str] = None
    in_response_to_tweet_id: Optional[str] = None
    brand: Optional[str] = None


class Message(BaseModel):
    tweet_id: str
    role: str  # "customer" | "agent"
    author_id: str
    text: str
    timestamp: str


class Conversation(BaseModel):
    conversation_id: str
    brand: str
    messages: List[Message]
    customer_author_id: str

    @property
    def turn_count(self) -> int:
        return len(self.messages)

    @property
    def is_multi_turn(self) -> bool:
        return len(self.messages) >= 2

    @property
    def initial_customer_message(self) -> Optional[str]:
        for msg in self.messages:
            if msg.role == "customer":
                return msg.text
        return None

    @property
    def first_agent_response(self) -> Optional[str]:
        for msg in self.messages:
            if msg.role == "agent":
                return msg.text
        return None


class BrandStats(BaseModel):
    brand_name: str
    total_tweets: int
    customer_messages: int
    agent_messages: int
    total_conversations: int
    avg_conversation_length: float
    multi_turn_conversations: int
    intent_diversity_score: float
    suitability_score: float


class DatasetStats(BaseModel):
    total_rows: int
    columns: List[str]
    missing_values: dict
    unique_brands: int
    unique_authors: int
    date_range: Optional[List[str]] = None
    unique_conversations: int
