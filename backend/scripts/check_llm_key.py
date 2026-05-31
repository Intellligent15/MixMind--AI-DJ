"""Standalone sanity-check for the configured LLM provider's API key.

Reads LLM_PROVIDER and the relevant *_API_KEY from backend/.env (same
path pydantic-settings uses), then makes one trivial generation call
against the right SDK to verify the credential. Does NOT touch the
DB, storage, or any project code — purely validates the key.

Usage:
    cd backend && uv run python scripts/check_llm_key.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _env(name: str) -> str | None:
    """Read a single key from backend/.env without loading app config."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


def _check_gemini() -> int:
    import google.genai as genai

    key = os.environ.get("GEMINI_API_KEY") or _env("GEMINI_API_KEY")
    if not key:
        print("FAIL: GEMINI_API_KEY is empty")
        return 1

    prefix = key[:6]
    print(f"Calling gemini-2.5-flash with key prefix {prefix!r}...")
    try:
        client = genai.Client(
            api_key=key,
            http_options=genai.types.HttpOptions(timeout=30_000),
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with exactly the word: pong",
        )
    except Exception as exc:
        msg = str(exc)
        print(f"FAIL: {type(exc).__name__}: {msg[:600]}")
        if "RESOURCE_EXHAUSTED" in msg and "'0'" in msg:
            print(
                "\nDiagnosis: project quota is zero. Either enable billing on "
                "the GCP project this key lives in (then create a NEW key — "
                "old ones can stay stuck), or switch to a different "
                "LLM_PROVIDER (groq is recommended)."
            )
        elif "401" in msg or "UNAUTHENTICATED" in msg:
            print("\nDiagnosis: the API rejected the credential as invalid.")
        elif "403" in msg or "PERMISSION_DENIED" in msg:
            print("\nDiagnosis: credential lacks permission for the "
                  "Generative Language API.")
        return 1

    text = (resp.text or "").strip()
    print(f"PASS: model returned: {text!r}")
    return 0


def _check_groq() -> int:
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY") or _env("GROQ_API_KEY")
    if not key:
        print("FAIL: GROQ_API_KEY is empty")
        return 1

    model = _env("GROQ_MODEL") or "openai/gpt-oss-120b"
    prefix = key[:6]
    print(f"Calling {model} with key prefix {prefix!r}...")
    try:
        client = Groq(api_key=key, timeout=30.0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
        )
    except Exception as exc:
        msg = str(exc)
        print(f"FAIL: {type(exc).__name__}: {msg[:600]}")
        lower = msg.lower()
        if "401" in msg or "invalid_api_key" in lower or "authentication" in lower:
            print("\nDiagnosis: API key invalid. Get one at "
                  "https://console.groq.com/keys")
        elif "429" in msg or "rate" in lower:
            print("\nDiagnosis: rate-limited. Free tier: 30 RPM / 1000 RPD.")
        elif "model" in lower and ("not found" in lower or "decommissioned" in lower):
            print(f"\nDiagnosis: model {model!r} unavailable. Try "
                  "'llama-3.3-70b-versatile' or see "
                  "https://console.groq.com/docs/models")
        return 1

    text = (resp.choices[0].message.content or "").strip()
    print(f"PASS: model returned: {text!r}")
    return 0


def main() -> int:
    provider = os.environ.get("LLM_PROVIDER") or _env("LLM_PROVIDER") or "gemini"
    print(f"LLM_PROVIDER={provider}")
    if provider == "gemini":
        return _check_gemini()
    if provider == "groq":
        return _check_groq()
    print(f"FAIL: unknown LLM_PROVIDER={provider!r} (expected gemini | groq)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
