import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm.groq import GroqProvider


@pytest.mark.asyncio
async def test_groq_provider_caching():
    """Cache hit returns persisted plan without calling the SDK."""
    mock_storage = AsyncMock()
    dummy_plan = [{"tool": "set_transition_window", "from_song_time_start": 1.0,
                   "to_song_time_start": 2.0, "duration_bars": 8}]
    mock_storage.exists.return_value = True
    mock_storage.read.return_value = json.dumps({"response": dummy_plan}).encode("utf-8")

    mock_client = MagicMock()
    mock_create = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("app.services.llm.groq.get_storage", return_value=mock_storage):
        with patch("app.services.llm.groq.Groq", return_value=mock_client):
            provider = GroqProvider(api_key="fake")
            plan = await provider.plan_transition(
                from_song={"bpm": 120},
                to_song={"bpm": 125},
                tools_schema="Tools desc",
            )

            assert plan == dummy_plan
            mock_create.assert_not_called()
            mock_storage.exists.assert_called_once()
            mock_storage.read.assert_called_once()


@pytest.mark.asyncio
async def test_groq_provider_generation_unwraps_plan_key():
    """Happy path: model returns {'plan': [...]}; provider unwraps to a list."""
    mock_storage = AsyncMock()
    mock_storage.exists.return_value = False

    dummy_plan = [{"tool": "set_transition_window", "from_song_time_start": 1.0,
                   "to_song_time_start": 2.0, "duration_bars": 8}]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({"plan": dummy_plan})))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("app.services.llm.groq.get_storage", return_value=mock_storage):
        with patch("app.services.llm.groq.Groq", return_value=mock_client):
            provider = GroqProvider(api_key="fake")
            plan = await provider.plan_transition(
                from_song={"bpm": 120},
                to_song={"bpm": 125},
                tools_schema="Tools desc",
            )

            assert plan == dummy_plan
            mock_client.chat.completions.create.assert_called_once()
            mock_storage.write.assert_called_once()
            written_key = mock_storage.write.call_args[0][0]
            assert written_key.startswith("mix_plan_logs/")
            written_data = json.loads(mock_storage.write.call_args[0][1].decode("utf-8"))
            assert written_data["response"] == dummy_plan
            assert written_data["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"


@pytest.mark.asyncio
async def test_groq_provider_accepts_raw_list():
    """If the model ignores the wrap instruction and returns a raw list,
    accept it instead of failing."""
    mock_storage = AsyncMock()
    mock_storage.exists.return_value = False

    dummy_plan = [{"tool": "set_transition_window", "from_song_time_start": 0.0,
                   "to_song_time_start": 0.0, "duration_bars": 4}]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(dummy_plan)))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("app.services.llm.groq.get_storage", return_value=mock_storage):
        with patch("app.services.llm.groq.Groq", return_value=mock_client):
            provider = GroqProvider(api_key="fake")
            plan = await provider.plan_transition({}, {}, "Tools desc")
            assert plan == dummy_plan


@pytest.mark.asyncio
async def test_groq_provider_invalid_json_raises():
    mock_storage = AsyncMock()
    mock_storage.exists.return_value = False

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="not json {{{"))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("app.services.llm.groq.get_storage", return_value=mock_storage):
        with patch("app.services.llm.groq.Groq", return_value=mock_client):
            provider = GroqProvider(api_key="fake")
            with pytest.raises(ValueError, match="Invalid JSON"):
                await provider.plan_transition({}, {}, "Tools desc")
            mock_storage.write.assert_not_called()


@pytest.mark.asyncio
async def test_groq_provider_unexpected_shape_raises():
    """Model returns an object without 'plan' key and not a list — reject."""
    mock_storage = AsyncMock()
    mock_storage.exists.return_value = False

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({"oops": "wrong shape"})))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("app.services.llm.groq.get_storage", return_value=mock_storage):
        with patch("app.services.llm.groq.Groq", return_value=mock_client):
            provider = GroqProvider(api_key="fake")
            with pytest.raises(ValueError, match="Expected JSON object"):
                await provider.plan_transition({}, {}, "Tools desc")
            mock_storage.write.assert_not_called()
