"""Local inference backend — works with any HuggingFace causal LM."""

from __future__ import annotations

import asyncio

import structlog

from visionnav.perception.fusion import Observation
from visionnav.settings import ModelSettings

log = structlog.get_logger(__name__)


class LocalModelBackend:
    def __init__(self, settings: ModelSettings) -> None:
        """
        Load local HuggingFace model and tokenizer.

        Why load once?
        Model loading is extremely expensive, so we initialize it once
        during backend startup instead of every prediction call.
        """

        log.info("loading_model", name=settings.name)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(settings.name)

        self._model = AutoModelForCausalLM.from_pretrained(
            settings.name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        self._model.eval()

        # DialoGPT has no pad token — reuse EOS token.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        log.info("model_ready", name=settings.name)

    async def predict_action(
        self,
        observation: Observation,
        task: str,
        history: list[dict],
        plan: list[str],
    ) -> str:
        """
        Generate the next agent action from the local language model.

        Why async?
        In real systems this may later become remote inference over HTTP,
        GPU workers, or distributed serving.
        """

        import torch

        text = (
            f"Task: {task}\n"
            f"Screen: {observation.to_text_summary()[:200]}\n"
            f"Action:"
        )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            ids = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        output = self._tokenizer.decode(
            ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Wrap output in parser-compatible action format.
        result = (
            f"<think>Model suggests: {output}</think>\n"
            f'<action>{{"type":"done","description":"{output[:50]}"}}</action>'
        )

        log.info("model_output", output=result[:100])

        return result

    async def predict_action_safe(
        self,
        observation: Observation,
        task: str,
        history: list[dict],
        plan: list[str],
        timeout_seconds: float = 30.0,
    ) -> str:
        """
        Calls predict_action with timeout protection.

        Why this matters:
        Models can hang because of:
        - GPU OOM
        - deadlocks
        - driver crashes
        - infinite generation
        - network failures

        Without timeout:
            agent waits forever

        With timeout:
            agent receives a valid FAIL action and continues safely

        Returns:
            Valid parser-compatible action string even on timeout.
        """

        try:
            result = await asyncio.wait_for(
                self.predict_action(
                    observation=observation,
                    task=task,
                    history=history,
                    plan=plan,
                ),
                timeout=timeout_seconds,
            )

            return result

        except asyncio.TimeoutError:
            log.warning(
                "model_timeout",
                timeout_seconds=timeout_seconds,
            )

            return (
                "<action>"
                f'{{"type":"fail","description":"Model timeout after '
                f'{timeout_seconds}s"}}'
                "</action>"
            )


# i just comment the belwo because here is i am using now casual lm but later i will go to qwen
# """Local HuggingFace Transformers backend — for development."""

# from __future__ import annotations
# import structlog
# from visionnav.perception.fusion import Observation
# from visionnav.settings import ModelSettings

# log = structlog.get_logger(__name__)


# class LocalModelBackend:

#     def __init__(self, settings: ModelSettings) -> None:
#         log.info("loading_model", name=settings.name)
#         import torch
#         from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

#         self._settings = settings
#         self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             settings.name,
#             torch_dtype=getattr(torch, settings.dtype, torch.bfloat16),
#             device_map=settings.device_map,
#         )
#         self._tokenizer = AutoTokenizer.from_pretrained(settings.name)
#         self._model.eval()
#         log.info("model_ready", name=settings.name)

#     async def predict_action(
#         self,
#         observation: Observation,
#         task: str,
#         history: list[dict],
#         plan: list[str],
#     ) -> str:
#         import torch

#         text = (
#             f"Task: {task}\n\n"
#             f"Screen text:\n{observation.to_text_summary()}\n\n"
#             f"What is the next action? Reply with "
#             f"<think>reasoning</think>"
#             f'<action>{{"type":"done","description":"result"}}</action>'
#         )

#         inputs = self._tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=2048,
#         ).to(self._model.device)

#         with torch.no_grad():
#             ids = self._model.generate(
#                 input_ids=inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 max_new_tokens=256,
#                 do_sample=False,
#                 pad_token_id=self._tokenizer.eos_token_id,
#             )

#         output = self._tokenizer.decode(
#             ids[0][inputs["input_ids"].shape[1] :],
#             skip_special_tokens=True,
#         )
#         log.info("model_output", output=output[:100])
#         return output
