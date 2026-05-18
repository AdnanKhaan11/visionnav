"""
vLLM HTTP client backend — for production high-throughput serving.

Start vLLM server first:
    make serve-vllm

Then set in .env:
    VISIONNAV_MODEL__BACKEND=vllm
    VISIONNAV_MODEL__VLLM__BASE_URL=http://localhost:8001
"""
from __future__ import annotations
import httpx
import structlog
from visionnav.models.prompt import build_prompt
from visionnav.perception.fusion import Observation
from visionnav.settings import VLLMSettings

log = structlog.get_logger(__name__)


class VLLMBackend:
    """
    PagedAttention + continuous batching = 10x throughput vs local inference.
    OpenAI-compatible API = swap providers with zero code changes.
    """

    def __init__(self, settings: VLLMSettings) -> None:
        self._settings = settings
        self._client   = httpx.AsyncClient(timeout=settings.timeout_seconds)
        log.info("vllm_ready", url=settings.base_url, model=settings.model_name)

    async def predict_action(
        self, observation: Observation, task: str,
        history: list[dict], plan: list[str],
    ) -> str:
        messages = build_prompt(observation, task, history, plan)
        resp     = await self._client.post(
            f"{self._settings.base_url}/v1/chat/completions",
            json={"model": self._settings.model_name, "messages": messages,
                  "max_tokens": 512, "temperature": 0.1},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
