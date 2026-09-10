import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrandConfig(BaseModel):
    default_brand: str = "AmazonHelp"


class PathsConfig(BaseModel):
    raw_data: str = "data/raw/twcs.csv"
    sample_data: str = "data/raw/sample_twcs.csv"
    processed_data_dir: str = "data/processed"
    golden_set: str = "data/golden/golden_set.jsonl"
    intent_taxonomy: str = "artifacts/intent_taxonomy.json"
    retrieval_index_dir: str = "artifacts/retrieval_index"
    evaluation_results: str = "artifacts/evaluation_results.json"
    cache_dir: str = "data/cache"


class SplitsConfig(BaseModel):
    dev_ratio: float = 0.70
    val_ratio: float = 0.15
    eval_ratio: float = 0.15
    seed: int = 42


class RetrievalConfig(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    similarity_metric: str = "cosine"


class PolicyConfig(BaseModel):
    auto_handle_intent_threshold: float = 0.80
    auto_handle_retrieval_threshold: float = 0.20
    min_evidence_count: int = 1
    sensitive_intents: List[str] = [
        "account_security",
        "complaint",
        "fraud_report",
        "legal_complaint"
    ]


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 300


class JudgeConfig(BaseModel):
    model: str = "gpt-4o-mini"
    sample_size: int = 50


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    BRAND_NAME: str = "AmazonHelp"
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K: int = 5
    AUTO_HANDLE_INTENT_THRESHOLD: float = 0.80
    AUTO_HANDLE_RETRIEVAL_THRESHOLD: float = 0.20
    RANDOM_SEED: int = 42

    brand: BrandConfig = BrandConfig()
    paths: PathsConfig = PathsConfig()
    splits: SplitsConfig = SplitsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    policy: PolicyConfig = PolicyConfig()
    llm: LLMConfig = LLMConfig()
    judge: JudgeConfig = JudgeConfig()

    @classmethod
    def load_settings(cls, config_path: str = "configs/config.yaml") -> "Settings":
        config_dict: Dict[str, Any] = {}
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

        settings = cls(**config_dict)
        # Environment variable overrides
        if settings.BRAND_NAME:
            settings.brand.default_brand = settings.BRAND_NAME
        if settings.LLM_MODEL:
            settings.llm.model = settings.LLM_MODEL
        if settings.EMBEDDING_MODEL:
            settings.retrieval.embedding_model = settings.EMBEDDING_MODEL
        if settings.TOP_K:
            settings.retrieval.top_k = settings.TOP_K
        if settings.AUTO_HANDLE_INTENT_THRESHOLD:
            settings.policy.auto_handle_intent_threshold = settings.AUTO_HANDLE_INTENT_THRESHOLD
        if settings.AUTO_HANDLE_RETRIEVAL_THRESHOLD:
            settings.policy.auto_handle_retrieval_threshold = settings.AUTO_HANDLE_RETRIEVAL_THRESHOLD
        if settings.RANDOM_SEED:
            settings.splits.seed = settings.RANDOM_SEED

        return settings


settings = Settings.load_settings()
