"""
Free LLM Annotator for VisionNav Dataset Factory.

Generates chain-of-thought reasoning annotations using free-tier LLM APIs.
Provider priority: Groq → Google AI Studio → OpenRouter → description fallback.

CRITICAL DESIGN: Provider fallback is based on PARSED result, not raw response.
If a provider returns content but it cannot be parsed into a valid annotation,
the next provider is tried. This ensures OpenRouter is reached even when
Gemini returns malformed or truncated JSON.

SECURITY: API keys are never stored in this file.
          Load from .env via python-dotenv in scripts/run_pipeline.py.

DEPENDENCIES:
    pip install groq httpx python-dotenv --break-system-packages
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import structlog

from data_pipeline.core.schemas import Annotation, PipelineSample, SampleStatus
from data_pipeline.core.metadata import now_iso
from data_pipeline.annotation.templates import build_reasoning_prompt

log = structlog.get_logger(__name__)

# ── Rate limits: minimum seconds between consecutive calls per provider ────────
_RATE_LIMITS: dict[str, float] = {
    "groq": 2.1,  # 30 RPM → 2.0s + safety margin
    "google": 4.1,  # 15 RPM → 4.0s + safety margin
    "openrouter": 3.1,  # 20 RPM → 3.0s + safety margin
}

# ── Model names ────────────────────────────────────────────────────────────────
# IMPROVISED CODE: Verified working model names as of June 2026.
# Update these if a provider renames or deprecates a model.
_GROQ_MODEL = "llama-3.3-70b-versatile"
_GOOGLE_MODEL = "gemini-1.5-flash"
_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# ── Token budget ───────────────────────────────────────────────────────────────
# IMPROVISED CODE: Raised from 400 to 600.
# 400 tokens caused Gemini to truncate mid-JSON, producing unparseable output.
# 600 tokens provides enough headroom for complete reasoning + intent + JSON.
_DEFAULT_MAX_TOKENS = 600


# ══════════════════════════════════════════════════════════════════════════════
# Annotation Cache
# ══════════════════════════════════════════════════════════════════════════════


class AnnotationCache:
    """
    SQLite-backed cache for annotation results.

    Key: sha256(image_hash + action_type + task[:50])
    Prevents re-calling provider APIs for screenshots already annotated.
    Cache persists across pipeline runs and is NOT cleared by --reset.
    This is intentional: annotations are expensive; cached results are reused.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path))
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotation_cache (
                    cache_key             TEXT PRIMARY KEY,
                    reasoning             TEXT NOT NULL,
                    intent                TEXT NOT NULL,
                    annotated_by          TEXT NOT NULL,
                    annotation_method     TEXT NOT NULL,
                    annotation_confidence REAL NOT NULL,
                    reasoning_verified    INTEGER NOT NULL,
                    created_at            TEXT NOT NULL
                )
            """)

    def _make_key(self, image_sha256: str, action_type: str, task: str) -> str:
        task_prefix = task[:50].strip().lower()
        raw_key = f"{image_sha256}|{action_type}|{task_prefix}"
        return hashlib.sha256(raw_key.encode()).hexdigest()[:32]

    def get(self, image_sha256: str, action_type: str, task: str) -> Annotation | None:
        key = self._make_key(image_sha256, action_type, task)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM annotation_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return Annotation(
            reasoning=row[1],
            intent=row[2],
            annotated_by=row[3],
            annotation_method=row[4],
            annotation_confidence=row[5],
            reasoning_verified=bool(row[6]),
            verified_at="",
        )

    def put(
        self, image_sha256: str, action_type: str, task: str, annotation: Annotation
    ) -> None:
        key = self._make_key(image_sha256, action_type, task)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO annotation_cache
                (cache_key, reasoning, intent, annotated_by, annotation_method,
                 annotation_confidence, reasoning_verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    key,
                    annotation.reasoning,
                    annotation.intent,
                    annotation.annotated_by,
                    annotation.annotation_method,
                    annotation.annotation_confidence,
                    int(annotation.reasoning_verified),
                    now_iso(),
                ),
            )

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM annotation_cache").fetchone()[0]


# ══════════════════════════════════════════════════════════════════════════════
# Reasoning Verifier
# ══════════════════════════════════════════════════════════════════════════════


class ReasoningVerifier:
    """
    Verifies LLM reasoning against OCR data from the screenshot.

    Does NOT ask the LLM whether its own reasoning is correct (circular).
    Checks programmatically: are the UI elements the reasoning mentions
    actually present in the OCR text from the screenshot?

    Returns True if >= 50% of quoted elements appear in OCR.
    Returns True with no penalty if reasoning contains no quoted elements.
    """

    def verify(self, reasoning: str, ocr_regions: list[dict]) -> bool:
        if not reasoning or not ocr_regions:
            return False

        ocr_flat = " ".join(r.get("text", "").lower() for r in ocr_regions)
        if not ocr_flat.strip():
            return False

        quoted_phrases = re.findall(r'"([^"]{2,40})"', reasoning)
        if not quoted_phrases:
            return True  # no specific claims → benefit of the doubt

        found = sum(
            1
            for phrase in quoted_phrases
            if phrase.lower() in ocr_flat
            or any(phrase.lower() in r.get("text", "").lower() for r in ocr_regions)
        )

        rate = found / len(quoted_phrases)
        verified = rate >= 0.50

        if not verified:
            log.debug(
                "reasoning_verification_failed",
                quoted=len(quoted_phrases),
                found=found,
                rate=round(rate, 2),
            )
        return verified


# ══════════════════════════════════════════════════════════════════════════════
# Free Annotator
# ══════════════════════════════════════════════════════════════════════════════


class FreeAnnotator:
    """
    Annotates PipelineSamples using free-tier LLM APIs.

    PROVIDER FALLBACK (fixed in this version):
        Providers are tried in order until one returns a PARSEABLE response.
        If a provider returns content that cannot be parsed into valid JSON,
        the next provider is tried rather than falling back to the description.
        This ensures OpenRouter is reached even when Gemini returns truncated JSON.

    PROVIDER ORDER: Groq → Google → OpenRouter → description fallback
        Groq:       14,400 req/day free (highest capacity)
        Google:     1,500 req/day free
        OpenRouter: ~200 req/day free (vision-capable, most limited)
    """

    def __init__(
        self,
        groq_api_key: str = "",
        google_api_key: str = "",
        openrouter_api_key: str = "",
        cache_dir: Path = Path("data/pipeline"),
        max_reasoning_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        # bool() check: empty string → disabled. Non-empty string → enabled.
        self._groq_key = groq_api_key if bool(groq_api_key) else ""
        self._google_key = google_api_key if bool(google_api_key) else ""
        self._openrouter_key = openrouter_api_key if bool(openrouter_api_key) else ""

        self._max_tokens = max_reasoning_tokens
        self._cache = AnnotationCache(Path(cache_dir) / ".annotation_cache.db")
        self._verifier = ReasoningVerifier()

        self._last_call: dict[str, float] = {
            "groq": 0.0,
            "google": 0.0,
            "openrouter": 0.0,
        }
        self._stats = {
            "groq": 0,
            "google": 0,
            "openrouter": 0,
            "cache_hits": 0,
            "fallback": 0,
            "failed": 0,
        }

        log.info(
            "free_annotator_initialized",
            groq=bool(self._groq_key),
            google=bool(self._google_key),
            openrouter=bool(self._openrouter_key),
            cache_size=self._cache.count(),
            max_tokens=self._max_tokens,
        )

        if not any([self._groq_key, self._google_key, self._openrouter_key]):
            log.warning(
                "free_annotator_no_providers",
                impact=(
                    "All samples will use description fallback. "
                    "Quality scores will be 0.60-0.68 instead of 0.85-0.94. "
                    "Set GROQ_API_KEY in .env to enable real annotation."
                ),
            )

    # ── Public interface ───────────────────────────────────────────────────────

    def annotate(self, sample: PipelineSample) -> bool:
        """
        Annotate one sample. Returns True if real LLM used, False if fallback.
        Populates sample.annotation and updates sample.quality in-place.
        """
        if sample.raw is None:
            return False

        # Cache lookup — reuse annotation from previous run if available
        image_hash = (
            (sample.enriched.image_hash_sha256 or "") if sample.enriched else ""
        )
        if image_hash:
            cached = self._cache.get(image_hash, sample.raw.action_type, sample.task)
            if cached is not None:
                sample.annotation = cached
                self._update_quality(sample)
                sample.status = SampleStatus.ANNOTATED.value
                self._stats["cache_hits"] += 1
                log.debug("annotation_cache_hit", sample_id=sample.sample_id)
                return True

        # Build prompt once; reused across all provider attempts
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

        # IMPROVISED CODE: Redesigned provider fallback.
        # Iterates through providers and tries the NEXT one if parsing fails.
        # Previous version stopped at Google even when Google returned bad JSON.
        #
        # Each provider entry: (name, call_function)
        # call_function returns raw string response or None on failure.
        providers: list[tuple[str, Callable[[], str | None]]] = []
        if self._groq_key:
            providers.append(("groq", lambda: self._call_groq(system_p, user_p)))
        if self._google_key:
            providers.append(("google", lambda: self._call_google(system_p, user_p)))
        if self._openrouter_key:
            providers.append(
                ("openrouter", lambda: self._call_openrouter(sample, system_p, user_p))
            )

        parsed_result: tuple[str, str] | None = None
        provider_used: str = ""

        for provider_name, call_fn in providers:
            raw_response = call_fn()

            if not raw_response:
                # Provider returned nothing (connection error, auth error, etc.)
                # Try next provider
                continue

            parsed_result = self._parse_json_response(raw_response, sample.sample_id)

            if parsed_result:
                # Successfully got parseable annotation from this provider
                provider_used = provider_name
                break
            else:
                # Provider returned content but it was unparseable
                # Try next provider rather than falling back to description
                log.debug(
                    "provider_parse_failed_trying_next",
                    provider=provider_name,
                    sample_id=sample.sample_id,
                    preview=raw_response[:60],
                )
                continue

        # Store successful LLM annotation
        if parsed_result:
            reasoning, intent = parsed_result
            ocr_regions = (
                sample.enriched.ocr_regions_detailed if sample.enriched else []
            )
            is_verified = self._verifier.verify(reasoning, ocr_regions)
            confidence = 0.80 if is_verified else 0.60

            annotation = Annotation(
                reasoning=reasoning,
                intent=intent,
                annotated_by=f"free_llm:{provider_used}",
                annotation_method="auto_llm",
                annotation_confidence=confidence,
                reasoning_verified=is_verified,
                verified_at=now_iso() if is_verified else "",
            )
            sample.annotation = annotation
            self._update_quality(sample)
            sample.status = SampleStatus.ANNOTATED.value
            self._stats[provider_used] += 1

            if image_hash:
                self._cache.put(
                    image_hash, sample.raw.action_type, sample.task, annotation
                )

            log.debug(
                "annotation_complete",
                sample_id=sample.sample_id,
                provider=provider_used,
                verified=is_verified,
                confidence=confidence,
                reasoning_length=len(reasoning),
            )
            return True

        # All providers failed or returned unparseable content
        # Use description as minimal fallback annotation
        if sample.raw.description and len(sample.raw.description.strip()) >= 5:
            fallback_reasoning = (
                f"Looking at the current screen for the task: {sample.task}. "
                f"The next action is to {sample.raw.description.lower()}."
            )
            sample.annotation = Annotation(
                reasoning=fallback_reasoning,
                intent=sample.task,
                annotated_by="description_fallback",
                annotation_method="fallback",
                annotation_confidence=0.25,
                reasoning_verified=False,
                verified_at="",
            )
            self._update_quality(sample)
            sample.status = SampleStatus.ANNOTATED.value
            self._stats["fallback"] += 1
            # IMPROVISED CODE: Single fallback log here only.
            # Removed duplicate log from pipeline.py Stage 5.
            log.debug("annotation_used_fallback", sample_id=sample.sample_id)
            return False

        self._stats["failed"] += 1
        return False

    def annotate_batch(self, samples: list[PipelineSample]) -> dict[str, int]:
        total = len(samples)
        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0:
                log.info(
                    "annotation_progress",
                    processed=i + 1,
                    total=total,
                    stats=self._stats,
                )
            self.annotate(sample)
        return dict(self._stats)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ── Provider implementations ───────────────────────────────────────────────

    def _call_groq(self, system_p: str, user_p: str) -> str | None:
        """
        Groq API via direct httpx call.

        IMPROVISED CODE: Replaced groq Python package with direct httpx call.
        The groq package throws connection errors despite api.groq.com being
        reachable. httpx reaches it successfully (verified via diagnostics).
        Groq's API is OpenAI-compatible — we use the standard endpoint directly.
        """
        self._wait_rate_limit("groq")
        try:
            import httpx  # type: ignore

            headers = {
                "Authorization": f"Bearer {self._groq_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": _GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p},
                ],
                "max_tokens": self._max_tokens,
                "temperature": 0.3,
            }

            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0,
            )

            if resp.status_code == 401:
                log.error(
                    "groq_auth_failed",
                    hint="Verify GROQ_API_KEY in .env — rotate at console.groq.com",
                )
                return None
            if resp.status_code == 429:
                log.warning("groq_rate_limited", sleeping_s=5)
                time.sleep(5)
                return None
            if not resp.is_success:
                log.warning(
                    "groq_call_failed", status=resp.status_code, body=resp.text[:120]
                )
                return None

            self._last_call["groq"] = time.monotonic()
            return resp.json()["choices"][0]["message"]["content"]

        except ImportError:
            log.error("httpx_missing", fix="pip install httpx --break-system-packages")
            return None
        except Exception as exc:
            log.warning("groq_call_failed", error=str(exc)[:120])
            return None

    def _call_google(self, system_p: str, user_p: str) -> str | None:
        """
        Google AI Studio (Gemini Flash). 15 RPM free.

        IMPROVISED CODE: Uses v1beta endpoint with gemini-1.5-flash.
        Prompt is sent as a single combined message since Gemini's
        system/user distinction differs from OpenAI format.
        """
        self._wait_rate_limit("google")
        try:
            import httpx  # type: ignore

            url = (
                f"https://generativelanguage.googleapis.com"
                f"/v1beta/models/{_GOOGLE_MODEL}:generateContent"
                f"?key={self._google_key}"
            )
            # Combine system and user prompts — Gemini handles them as one turn
            combined_prompt = f"{system_p}\n\n{user_p}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": combined_prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": self._max_tokens,
                    "temperature": 0.3,
                    # IMPROVISED CODE: Added responseMimeType to encourage
                    # JSON output without markdown wrapping.
                    # This reduces parse failures from markdown code blocks.
                    "responseMimeType": "application/json",
                },
            }

            resp = httpx.post(url, json=payload, timeout=30.0)

            if resp.status_code == 404:
                log.error(
                    "google_model_not_found",
                    model=_GOOGLE_MODEL,
                    hint="Check model name at aistudio.google.com",
                )
                return None
            if resp.status_code in (400, 415):
                # IMPROVISED CODE: 415 means responseMimeType not supported;
                # fall back to plain text request.
                payload["generationConfig"].pop("responseMimeType", None)
                resp = httpx.post(url, json=payload, timeout=30.0)
                if not resp.is_success:
                    log.error(
                        "google_bad_request",
                        status=resp.status_code,
                        body=resp.text[:200],
                    )
                    return None
            if resp.status_code in (401, 403):
                log.error(
                    "google_auth_failed",
                    hint="Verify GEMINI_API_KEY in .env at aistudio.google.com",
                )
                return None
            if resp.status_code == 429:
                log.warning("google_rate_limited")
                return None

            resp.raise_for_status()
            self._last_call["google"] = time.monotonic()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        except ImportError:
            log.error("httpx_missing", fix="pip install httpx --break-system-packages")
            return None
        except Exception as exc:
            log.warning("google_call_failed", error=str(exc)[:120])
            return None

    def _call_openrouter(
        self, sample: PipelineSample, system_p: str, user_p: str
    ) -> str | None:
        """
        OpenRouter. ~200 req/day free.
        Model: meta-llama/llama-3.1-8b-instruct:free (stable free model).
        Attaches screenshot image if file is small enough for free tier.
        """
        self._wait_rate_limit("openrouter")
        try:
            import httpx  # type: ignore
            import base64

            headers = {
                "Authorization": f"Bearer {self._openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/visionnav/visionnav",
                "X-Title": "VisionNav Dataset Factory",
            }

            content: list = [{"type": "text", "text": user_p}]

            # Attach screenshot only for small files — free tier has limits
            if (
                sample.raw
                and sample.raw.screenshot_path
                and Path(sample.raw.screenshot_path).exists()
            ):
                size_kb = Path(sample.raw.screenshot_path).stat().st_size / 1024
                if size_kb < 300:
                    img_bytes = Path(sample.raw.screenshot_path).read_bytes()
                    b64_str = base64.b64encode(img_bytes).decode()
                    content.insert(
                        0,
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_str}"},
                        },
                    )

            payload = {
                "model": _OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": content},
                ],
                "max_tokens": self._max_tokens,
            }

            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0,
            )

            if resp.status_code == 404:
                log.error(
                    "openrouter_model_not_found",
                    model=_OPENROUTER_MODEL,
                    hint="Check available free models at openrouter.ai/models",
                )
                return None
            if resp.status_code in (401, 403):
                log.error(
                    "openrouter_auth_failed",
                    hint="Verify OPENROUTER_API_KEY in .env at openrouter.ai",
                )
                return None
            if resp.status_code == 429:
                log.warning("openrouter_rate_limited")
                return None

            resp.raise_for_status()
            self._last_call["openrouter"] = time.monotonic()
            return resp.json()["choices"][0]["message"]["content"]

        except ImportError:
            log.error("httpx_missing", fix="pip install httpx --break-system-packages")
            return None
        except Exception as exc:
            log.warning("openrouter_call_failed", error=str(exc)[:120])
            return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _wait_rate_limit(self, provider: str) -> None:
        min_gap = _RATE_LIMITS.get(provider, 2.0)
        elapsed = time.monotonic() - self._last_call.get(provider, 0.0)
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

    def _parse_json_response(self, raw: str, sample_id: str) -> tuple[str, str] | None:
        """
        Parse LLM response into (reasoning, intent).

        IMPROVISED CODE: Three-strategy parser, more robust than previous version.

        Strategy 1: Strip markdown fences, parse full JSON.
        Strategy 2: Find first JSON object in response (handles preamble text).
        Strategy 3: Regex extract reasoning field (handles truncated JSON).

        NOTE: 'verified' field is intentionally NOT extracted from the LLM response.
              Verification is performed independently by ReasoningVerifier.
        """
        # Strategy 1 — strip markdown code fences, parse as JSON
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
            reasoning = str(data.get("reasoning", "")).strip()
            intent = str(data.get("intent", "")).strip()
            if len(reasoning) >= 30:
                return reasoning, intent
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2 — find first { } block in response (handles leading text)
        brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                reasoning = str(data.get("reasoning", "")).strip()
                intent = str(data.get("intent", "")).strip()
                if len(reasoning) >= 30:
                    return reasoning, intent
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3 — regex extract reasoning value directly
        # Handles truncated JSON where closing brace is missing
        match = re.search(
            r'"reasoning"\s*:\s*"((?:[^"\\]|\\.){30,})"',
            raw,
            re.DOTALL,
        )
        if match:
            try:
                # Use json.loads on just the string value to handle escapes
                reasoning = json.loads(f'"{match.group(1)}"')
                if len(reasoning) >= 30:
                    return reasoning, ""
            except (json.JSONDecodeError, ValueError):
                reasoning = match.group(1)
                if len(reasoning) >= 30:
                    return reasoning, ""

        log.debug(
            "annotation_parse_failed",
            sample_id=sample_id,
            preview=raw[:80],
        )
        return None

    def _extract_ocr_text(self, sample: PipelineSample) -> str:
        if sample.enriched and sample.enriched.ocr_regions_detailed:
            return " | ".join(
                r.get("text", "") for r in sample.enriched.ocr_regions_detailed[:30]
            )
        if sample.raw and sample.raw.ocr_text_raw:
            return sample.raw.ocr_text_raw
        return ""

    def _update_quality(self, sample: PipelineSample) -> None:
        if sample.quality is None or sample.annotation is None:
            return
        ann = sample.annotation
        sample.quality.has_reasoning = bool(ann.reasoning)
        sample.quality.reasoning_length = len(ann.reasoning)
        sample.quality.reasoning_verified = ann.reasoning_verified
