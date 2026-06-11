"""LLM provider factory.

Provider modules are imported lazily inside the factory so importing
`app.services.llm` (e.g. for the pure `prompts` module) doesn't drag in
all three vendor SDKs — only the configured provider's SDK is needed at
call time.
"""

from app.core.config import settings


def get_llm_provider():
    t = settings.llm_temperature
    if settings.llm_provider == "gemini":
        from app.services.llm.gemini import GeminiProvider

        return GeminiProvider(api_key=settings.gemini_api_key, temperature=t)
    if settings.llm_provider == "groq":
        from app.services.llm.groq import GroqProvider

        return GroqProvider(
            api_key=settings.groq_api_key, model=settings.groq_model, temperature=t
        )
    if settings.llm_provider == "digitalocean":
        from app.services.llm.digitalocean import DigitalOceanProvider

        return DigitalOceanProvider(
            api_key=settings.do_inference_api_key,
            model=settings.do_inference_model,
            temperature=t,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
