from typing import Protocol

class LLMProvider(Protocol):
    async def plan_transition(
        self,
        from_song: dict,
        to_song: dict,
        tools_schema: str,
    ) -> list[dict]: ...
