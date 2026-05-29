"""
Auto-annotator — generates reasoning using the Anthropic Claude API.

Why Claude for annotation?
  - Best-in-class instruction following
  - Strong structured output (JSON)
  - Excellent at visual reasoning descriptions
  - Available via the same API you already use

Architecture:
  PipelineSample
    → build prompt (templates.py)
    → call Claude API
    → parse JSON response
    → cross-reference reasoning with OCR
    → store in sample.annotation

Cross-reference check:
  Every UI element mentioned in reasoning must exist in OCR text.
  If reasoning says "click Submit" but OCR has no "Submit" → flag.
  This prevents hallucinated reasoning from entering training data.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import structlog

from data_pipeline.core.exceptions import AnnotationError
from data_pipeline.core.metadata import now_iso
from data_pipeline.core.schemas import Annotation, PipelineSample, SampleStatus
from data_pipeline.annotation.templates import build_reasoning_prompt

log = structlog.get_logger(__name__)

# Claude model to use for annotation
# Use claude-haiku-4-5 for cost, claude-sonnet-4-6 for quality
_ANNOTATION_MODEL = "claude-haiku-4-5-20251001"

# Max tokens for annotation response (reasoning is short)
_MAX_TOKENS = 400

# Retry configuration for API calls
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds between retries


@dataclass
class AnnotationResult:
    """Result of annotating one sample."""

    sample_id: str
    success: bool
    reasoning: str = ""
    intent: str = ""
    verified: bool = False
    error: str = ""
    tokens_used: int = 0
    model_used: str = ""


class AutoAnnotator:
    """
    Generates chain-of-thought reasoning for PipelineSamples.

    Cost estimation:
      claude-haiku-4-5:  ~$0.001 per sample
      claude-sonnet-4-6: ~$0.005 per sample
      For 100,000 samples: ~$100 - $500

    Usage:
        annotator = AutoAnnotator(api_key="sk-ant-...")
        result    = annotator.annotate(sample)

    Batch usage:
        results = annotator.annotate_batch(samples, delay_between=0.1)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _ANNOTATION_MODEL,
        max_retries: int = _MAX_RETRIES,
        retry_delay: float = _RETRY_DELAY,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._api_key = api_key
        self._call_count = 0
        self._total_tokens = 0

    def annotate(self, sample: PipelineSample) -> AnnotationResult:
        """
        Generate reasoning annotation for one sample.

        Stores result directly in sample.annotation.
        Updates sample.status to ANNOTATED on success.

        Args:
            sample: enriched PipelineSample (enrichment should be complete)

        Returns:
            AnnotationResult with success/failure details
        """
        if sample.raw is None:
            return AnnotationResult(
                sample_id=sample.sample_id,
                success=False,
                error="sample.raw is None — cannot annotate",
            )

        sample.status = SampleStatus.ANNOTATING.value

        # ── Build prompt ───────────────────────────────────────────────────
        ocr_text = self._extract_ocr_text(sample)

        system_prompt, user_prompt = build_reasoning_prompt(
            task=sample.task,
            ocr_text=ocr_text,
            action_type=sample.raw.action_type,
            step_index=sample.step_index,
            total_steps=self._estimate_total_steps(sample),
            description=sample.raw.description or "",
            coordinates=sample.raw.coordinates,
            text=sample.raw.text,
        )

        # ── Call API with retry ────────────────────────────────────────────
        raw_response = self._call_with_retry(system_prompt, user_prompt)
        if raw_response is None:
            sample.status = SampleStatus.ENRICHED.value  # revert
            return AnnotationResult(
                sample_id=sample.sample_id,
                success=False,
                error=f"API call failed after {self._max_retries} retries",
            )

        # ── Parse response ─────────────────────────────────────────────────
        parsed = self._parse_response(raw_response, sample.sample_id)
        if parsed is None:
            sample.status = SampleStatus.ENRICHED.value
            return AnnotationResult(
                sample_id=sample.sample_id,
                success=False,
                error=f"Failed to parse JSON response: {raw_response[:100]}",
            )

        reasoning, intent, verified = parsed

        # ── Cross-reference check ──────────────────────────────────────────
        # Verify that elements mentioned in reasoning exist in OCR text
        verification = self._verify_reasoning(reasoning, ocr_text)
        if not verification and verified:
            # Model claimed verified but elements not found in OCR
            log.warning(
                "reasoning_verification_mismatch",
                sample_id=sample.sample_id,
                reasoning_preview=reasoning[:80],
            )
            verified = False

        # ── Store annotation ───────────────────────────────────────────────
        sample.annotation = Annotation(
            reasoning=reasoning,
            intent=intent,
            annotated_by=self._model,
            annotation_method="auto_llm",
            annotation_confidence=0.8 if verified else 0.5,
            reasoning_verified=verified,
            verified_at=now_iso() if verified else "",
        )

        sample.status = SampleStatus.ANNOTATED.value
        sample.updated_at = now_iso()

        if sample.quality:
            sample.quality.has_reasoning = True
            sample.quality.reasoning_length = len(reasoning)
            sample.quality.reasoning_verified = verified

        log.info(
            "sample_annotated",
            sample_id=sample.sample_id,
            verified=verified,
            reasoning_len=len(reasoning),
        )

        return AnnotationResult(
            sample_id=sample.sample_id,
            success=True,
            reasoning=reasoning,
            intent=intent,
            verified=verified,
            model_used=self._model,
        )

    def annotate_batch(
        self,
        samples: list[PipelineSample],
        delay_between_s: float = 0.1,
    ) -> list[AnnotationResult]:
        """
        Annotate a batch of samples with rate limiting.

        Args:
            samples:         samples to annotate
            delay_between_s: seconds to wait between API calls
                             (avoid rate limiting)

        Returns:
            List of AnnotationResult in same order as input
        """
        results = []
        total = len(samples)

        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0:
                log.info(
                    "annotation_progress",
                    processed=i + 1,
                    total=total,
                    total_tokens=self._total_tokens,
                )

            result = self.annotate(sample)
            results.append(result)

            # Rate limiting — be respectful to the API
            if i < total - 1:
                time.sleep(delay_between_s)

        successes = sum(1 for r in results if r.success)
        log.info(
            "annotation_batch_complete",
            total=total,
            successes=successes,
            failures=total - successes,
            total_tokens=self._total_tokens,
        )

        return results

    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str | None:
        """
        Call Claude API with exponential backoff retry.
        Returns raw text response or None on permanent failure.
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        for attempt in range(1, self._max_retries + 1):
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                self._call_count += 1
                self._total_tokens += (
                    response.usage.input_tokens + response.usage.output_tokens
                )

                return response.content[0].text

            except Exception as exc:
                log.warning(
                    "api_call_failed",
                    attempt=attempt,
                    max=self._max_retries,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    wait = self._retry_delay * (
                        2 ** (attempt - 1)
                    )  # exponential backoff
                    time.sleep(wait)
                else:
                    log.error("api_call_exhausted_retries", error=str(exc))
                    return None

        return None

    def _parse_response(
        self,
        raw: str,
        sample_id: str,
    ) -> tuple[str, str, bool] | None:
        """
        Parse JSON response from Claude.
        Returns (reasoning, intent, verified) or None on parse failure.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = cleaned.replace("```", "").strip()

        try:
            data = json.loads(cleaned)
            reasoning = str(data.get("reasoning", "")).strip()
            intent = str(data.get("intent", "")).strip()
            verified = bool(data.get("verified", False))

            if not reasoning:
                log.warning("empty_reasoning", sample_id=sample_id)
                return None

            return reasoning, intent, verified

        except json.JSONDecodeError as exc:
            log.warning(
                "json_parse_failed",
                sample_id=sample_id,
                error=str(exc),
                raw=raw[:100],
            )
            return None

    def _verify_reasoning(
        self,
        reasoning: str,
        ocr_text: str,
    ) -> bool:
        """
        Cross-reference: check that key noun phrases in reasoning
        actually appear in the OCR text.

        This is a lightweight check — not perfect, but catches
        obvious hallucinations like "I click the Submit button"
        when no "Submit" appears on screen.

        Returns True if reasoning passes verification.
        """
        if not ocr_text or not reasoning:
            return False

        ocr_lower = ocr_text.lower()
        reasoning_lower = reasoning.lower()

        # Extract quoted phrases from reasoning (agent says "X button")
        quoted = re.findall(r'"([^"]+)"', reasoning_lower)
        for phrase in quoted:
            if len(phrase) > 2 and phrase not in ocr_lower:
                return False  # quoted phrase not found in OCR

        return True

    def _extract_ocr_text(self, sample: PipelineSample) -> str:
        """Get best available OCR text for the sample."""
        if sample.enriched and sample.enriched.ocr_regions_detailed:
            return " ".join(r["text"] for r in sample.enriched.ocr_regions_detailed)
        if sample.raw and sample.raw.ocr_text_raw:
            return sample.raw.ocr_text_raw
        return ""

    def _estimate_total_steps(self, sample: PipelineSample) -> int:
        """
        Rough estimate of total steps in this trajectory.
        Used in prompt context to help model understand task scope.
        """
        # We don't have full trajectory context at sample level
        # Use step_index as lower bound
        return max(sample.step_index + 1, 5)

    @property
    def stats(self) -> dict:
        """Return annotation statistics."""
        return {
            "calls": self._call_count,
            "tokens": self._total_tokens,
            "model": self._model,
        }
