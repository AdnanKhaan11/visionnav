"""
Free LLM Annotator using Groq + OpenRouter fallback.

Cost: $0
Daily capacity: ~3,000 samples (Groq primary + OpenRouter fallback)

Provider priority:
  1. Groq (llama-3.3-70b) — fastest, text-only, 14.4K req/day
  2. OpenRouter (llama-4-maverick:free) — vision-capable, 200 req/day
  3. Google AI Studio (gemini-flash) — backup, 1,500 req/day
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum

import structlog

from data_pipeline.core.schemas import Annotation, PipelineSample, SampleStatus
from data_pipeline.core.metadata import now_iso
from data_pipeline.annotation.templates import build_reasoning_prompt

log = structlog.get_logger(__name__)


class Provider(Enum):
    GROQ = "groq"
    OPENROUTER = "openrouter"
    GOOGLE = "google"


@dataclass
class AnnotationConfig:
    """
    Configuration for free-tier annotation.
    All providers are free — no payment required.
    """

    # Groq (primary — fastest, text-only)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # OpenRouter (secondary — vision capable, free tier)
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-4-maverick:free"

    # Google AI Studio (fallback)
    google_api_key: str = ""
    google_model: str = "gemini-1.5-flash"

    # Behavior
    delay_between_s: float = 0.5  # respect rate limits
    max_retries: int = 2


class FreeAnnotator:
    """
    Annotates PipelineSamples using free LLM APIs.

    Tries providers in order: Groq → OpenRouter → Google.
    If all fail, uses description-based fallback (no API call needed).

    Zero cost. Zero credit card. ~3,000 samples/day capacity.

    Get API keys:
        Groq:       console.groq.com (no credit card)
        OpenRouter: openrouter.ai    (no credit card)
        Google:     aistudio.google.com (no credit card)
    """

    def __init__(self, config: AnnotationConfig) -> None:
        self._cfg = config
        self._call_counts = {p: 0 for p in Provider}

    def annotate(self, sample: PipelineSample) -> bool:
        """
        Generate reasoning annotation using first available free provider.

        Returns True if annotation was added, False if fallback was used.
        """
        if sample.raw is None:
            return False

        # Build prompt
        ocr_text = self._extract_ocr_text(sample)
        system_p, user_p = build_reasoning_prompt(
            task=sample.task,
            ocr_text=ocr_text,
            action_type=sample.raw.action_type,
            step_index=sample.step_index,
            total_steps=max(sample.step_index + 1, 5),
            description=sample.raw.description or "",
            coordinates=sample.raw.coordinates,
            text=sample.raw.text,
        )

        # Try each provider in order
        result = None

        if self._cfg.groq_api_key:
            result = self._call_groq(system_p, user_p)

        if result is None and self._cfg.openrouter_api_key:
            result = self._call_openrouter(sample, system_p, user_p)

        if result is None and self._cfg.google_api_key:
            result = self._call_google(system_p, user_p)

        # Parse and store result
        if result:
            parsed = self._parse_response(result, sample.sample_id)
            if parsed:
                reasoning, intent, verified = parsed
                sample.annotation = Annotation(
                    reasoning=reasoning,
                    intent=intent,
                    annotated_by="free_llm",
                    annotation_method="auto_llm",
                    annotation_confidence=0.75 if verified else 0.55,
                    reasoning_verified=verified,
                    verified_at=now_iso() if verified else "",
                )
                if sample.quality:
                    sample.quality.has_reasoning = True
                    sample.quality.reasoning_length = len(reasoning)
                    sample.quality.reasoning_verified = verified
                sample.status = SampleStatus.ANNOTATED.value
                return True

        # Fallback: use description as minimal annotation (no API needed)
        if sample.raw.description:
            sample.annotation = Annotation(
                reasoning=(
                    f"Looking at the screen for task: {sample.task}. "
                    f"The next action is: {sample.raw.description}."
                ),
                intent=sample.task,
                annotated_by="description_fallback",
                annotation_method="fallback",
                annotation_confidence=0.30,
            )
            if sample.quality:
                sample.quality.has_reasoning = True
                sample.quality.reasoning_length = len(sample.annotation.reasoning)
            sample.status = SampleStatus.ANNOTATED.value

        return False  # used fallback, not real LLM

    def annotate_batch(
        self,
        samples: list[PipelineSample],
        delay_between_s: float | None = None,
    ) -> dict[str, int]:
        """
        Annotate a batch. Returns counts by method.
        """
        delay = delay_between_s or self._cfg.delay_between_s
        counts = {"llm": 0, "fallback": 0, "failed": 0}
        total = len(samples)

        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                log.info("annotation_progress", done=i + 1, total=total, counts=counts)

            try:
                used_llm = self.annotate(sample)
                counts["llm" if used_llm else "fallback"] += 1
            except Exception as exc:
                log.error(
                    "annotation_error", sample_id=sample.sample_id, error=str(exc)
                )
                counts["failed"] += 1

            if i < total - 1:
                time.sleep(delay)

        log.info("annotation_batch_complete", **counts)
        return counts

    # ── Provider implementations ───────────────────────────────────────────

    def _call_groq(self, system_p: str, user_p: str) -> str | None:
        """Call Groq API. Returns raw text or None on failure."""
        try:
            from groq import Groq

            client = Groq(api_key=self._cfg.groq_api_key)
            response = client.chat.completions.create(
                model=self._cfg.groq_model,
                messages=[
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p},
                ],
                max_tokens=400,
                temperature=0.3,
            )
            self._call_counts[Provider.GROQ] += 1
            return response.choices[0].message.content
        except Exception as exc:
            log.warning("groq_call_failed", error=str(exc)[:80])
            return None

    def _call_openrouter(
        self,
        sample: PipelineSample,
        system_p: str,
        user_p: str,
    ) -> str | None:
        """
        Call OpenRouter with optional image attachment.
        Llama-4-Maverick:free supports vision — attach screenshot for better reasoning.
        """
        try:
            import httpx, base64
            from pathlib import Path

            headers = {
                "Authorization": f"Bearer {self._cfg.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/visionnav",
            }

            # Build content — include screenshot if available
            content: list = [{"type": "text", "text": user_p}]

            if (
                sample.raw
                and sample.raw.screenshot_path
                and Path(sample.raw.screenshot_path).exists()
            ):
                img_data = Path(sample.raw.screenshot_path).read_bytes()
                b64 = base64.b64encode(img_data).decode()
                content.insert(
                    0,
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                )

            payload = {
                "model": self._cfg.openrouter_model,
                "messages": [
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 400,
            }

            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            self._call_counts[Provider.OPENROUTER] += 1
            return resp.json()["choices"][0]["message"]["content"]

        except Exception as exc:
            log.warning("openrouter_call_failed", error=str(exc)[:80])
            return None

    def _call_google(self, system_p: str, user_p: str) -> str | None:
        """Call Google AI Studio (Gemini Flash — 1,500 req/day free)."""
        try:
            import httpx

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self._cfg.google_model}:generateContent"
                f"?key={self._cfg.google_api_key}"
            )
            payload = {
                "contents": [{"parts": [{"text": f"{system_p}\n\n{user_p}"}]}],
                "generationConfig": {"maxOutputTokens": 400, "temperature": 0.3},
            }
            resp = httpx.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
            self._call_counts[Provider.GOOGLE] += 1
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            log.warning("google_call_failed", error=str(exc)[:80])
            return None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str, sample_id: str) -> tuple[str, str, bool] | None:
        """Parse JSON from LLM response."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            reasoning = str(data.get("reasoning", "")).strip()
            intent = str(data.get("intent", "")).strip()
            verified = bool(data.get("verified", False))
            if len(reasoning) > 20:
                return reasoning, intent, verified
        except json.JSONDecodeError:
            pass
        return None

    def _extract_ocr_text(self, sample: PipelineSample) -> str:
        if sample.enriched and sample.enriched.ocr_regions_detailed:
            return " | ".join(
                r["text"] for r in sample.enriched.ocr_regions_detailed[:30]
            )
        return sample.raw.ocr_text_raw if sample.raw else ""

    @property
    def stats(self) -> dict:
        return {p.value: n for p, n in self._call_counts.items()}
