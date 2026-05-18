"""Local HuggingFace Transformers backend — for development."""
from __future__ import annotations
import structlog
from visionnav.models.prompt import build_prompt
from visionnav.perception.fusion import Observation
from visionnav.settings import ModelSettings

log = structlog.get_logger(__name__)


class LocalModelBackend:
    """
    Loads Qwen2.5-VL into memory via HuggingFace.
    One inference at a time — good for dev/local runs.
    Switch to VLLMBackend for production.
    """

    def __init__(self, settings: ModelSettings) -> None:
        log.info("loading_model", name=settings.name)
        import torch
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        self._settings  = settings
        self._model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            settings.name,
            torch_dtype=getattr(torch, settings.dtype, torch.bfloat16),
            device_map=settings.device_map,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(settings.name)
        self._model.eval()
        log.info("model_ready", name=settings.name)

    async def predict_action(
        self, observation: Observation, task: str,
        history: list[dict], plan: list[str],
    ) -> str:
        import torch
        messages = build_prompt(observation, task, history, plan)
        inputs   = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs, max_new_tokens=512, temperature=0.1, do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(ids[0], skip_special_tokens=True)
