import os

from .base import LLM, Schema
from .gemini import GeminiLLM
from .groq import GroqLLM
from .openai import OpenAILLM

REGISTRY: dict[str, type[LLM]] = {
    "gemini": GeminiLLM,
    "groq": GroqLLM,
    "openai": OpenAILLM,
}


def get_llm(name: str = "gemini") -> LLM:
    if name not in REGISTRY:
        raise ValueError(f"Unknown LLM: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()


def get_answer_llm() -> LLM:
    """Return the LLM to use for RAG answer generation.
    Configured via AMB_ANSWER_LLM (provider) and AMB_ANSWER_MODEL (optional model override)."""
    provider = os.environ.get("AMB_ANSWER_LLM", "groq")
    model = os.environ.get("AMB_ANSWER_MODEL")
    cls = REGISTRY.get(provider)
    if cls is None:
        raise ValueError(f"Unknown AMB_ANSWER_LLM: '{provider}'. Available: {list(REGISTRY)}")
    return cls(model) if model else cls()


def get_judge_llm() -> LLM:
    """Return the LLM to use for evaluation/judging.
    Configured via AMB_JUDGE_LLM (provider) and AMB_JUDGE_MODEL (optional model override)."""
    provider = os.environ.get("AMB_JUDGE_LLM", "gemini")
    model = os.environ.get("AMB_JUDGE_MODEL")
    cls = REGISTRY.get(provider)
    if cls is None:
        raise ValueError(f"Unknown AMB_JUDGE_LLM: '{provider}'. Available: {list(REGISTRY)}")
    return cls(model) if model else cls()
