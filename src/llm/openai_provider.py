import os
import json
from typing import Optional
from src.llm.base import LLMProvider, LLMResponse
from src.llm.cache import DiskCache

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", cache_dir: str = "data/cache"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.cache = DiskCache(cache_dir=cache_dir)
        if OpenAI is not None and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 300,
        json_output: bool = False
    ) -> LLMResponse:
        cache_key = f"{self.model}:{system_prompt}:{prompt}:{temperature}:{json_output}"
        cached = self.cache.get(cache_key)
        if cached:
            return LLMResponse(**cached)

        if not self.client:
            raise RuntimeError("OpenAI API Key is missing or openai package is not installed.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        usage = response.usage

        res = LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0
        )
        self.cache.set(cache_key, res.model_dump())
        return res
