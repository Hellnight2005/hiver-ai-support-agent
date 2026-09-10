from src.llm.base import LLMProvider, LLMResponse
from src.llm.openai_provider import OpenAIProvider
from src.llm.mock_provider import MockProvider
from src.config import settings


def get_llm_provider(force_mock: bool = False) -> LLMProvider:
    if force_mock or not settings.OPENAI_API_KEY:
        return MockProvider(model="mock-provider")
    return OpenAIProvider(
        api_key=settings.OPENAI_API_KEY,
        model=settings.llm.model,
        cache_dir=settings.paths.cache_dir
    )
