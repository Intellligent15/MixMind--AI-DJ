from app.services.llm.provider import LLMProvider
from app.services.llm.gemini import GeminiProvider
from app.services.llm.groq import GroqProvider
from app.core.config import settings

def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "gemini":
        return GeminiProvider(api_key=settings.gemini_api_key)
    if settings.llm_provider == "groq":
        return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
