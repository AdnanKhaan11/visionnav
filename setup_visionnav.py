#!/usr/bin/env python3
"""
VisionNav Project Structure Setup Script
========================================
Run this script INSIDE your already-created visionnav/ repo folder.

Usage:
    cd visionnav          ← your cloned GitHub repo
    python setup_visionnav.py

What it does:
    - Creates every folder in the VisionNav MVP architecture
    - Creates every file with proper starter content
    - Never overwrites files that already exist (safe to re-run)
    - Prints a full summary at the end

Author: VisionNav Engineering
"""

import os
import sys
from pathlib import Path

# ─── COLOUR OUTPUT ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def p(msg: str, colour: str = "") -> None:
    print(f"{colour}{msg}{RESET}")


# ─── COUNTERS ─────────────────────────────────────────────────────────────────
created_dirs = 0
created_files = 0
skipped_files = 0


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def mkdir(path: Path) -> None:
    global created_dirs
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        created_dirs += 1


def write(path: Path, content: str = "") -> None:
    global created_files, skipped_files
    if path.exists():
        skipped_files += 1
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.lstrip("\n"), encoding="utf-8")
    created_files += 1
    p(f"  ✓  {path}", GREEN)


# ══════════════════════════════════════════════════════════════════════════════
#  FILE CONTENTS
# ══════════════════════════════════════════════════════════════════════════════

# ─── ROOT FILES ───────────────────────────────────────────────────────────────

GITIGNORE = """
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg
*.egg-info/
dist/
build/
.eggs/
.venv/
venv/
env/

# Environment
.env
.env.local
.env.*.local

# uv
.uv/

# Model weights (use HuggingFace Hub or DVC instead)
*.bin
*.safetensors
*.gguf
*.ckpt
checkpoints/
models/

# Data (use DVC)
data/raw/
data/processed/
data/augmented/
data/splits/
data/instruction_tuning/
training_data/

# Screenshots (runtime artifacts)
screenshots/
/tmp/visionnav/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage.xml

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Logs
*.log
logs/

# DVC
.dvc/cache
.dvc/tmp

# W&B
wandb/
"""

PYTHON_VERSION = "3.12\n"

ENV_EXAMPLE = """
# ============================================================
# VisionNav Environment Variables
# Copy this file to .env and fill in your real values.
# NEVER commit .env to Git.
# ============================================================

# Environment
VISIONNAV_ENV=development

# ── Model Backend ─────────────────────────────────────────
# Options: "local" | "vllm"
# Use "local" for dev (no GPU server needed)
# Use "vllm" for production (fast, scalable)
VISIONNAV_MODEL__BACKEND=local
VISIONNAV_MODEL__NAME=Qwen/Qwen2.5-VL-3B-Instruct
VISIONNAV_MODEL__DTYPE=bfloat16
VISIONNAV_MODEL__DEVICE_MAP=auto

# ── vLLM Server (only needed when backend=vllm) ────────────
VISIONNAV_MODEL__VLLM__BASE_URL=http://localhost:8001
VISIONNAV_MODEL__VLLM__MODEL_NAME=visionnav-3b
VISIONNAV_MODEL__VLLM__TIMEOUT_SECONDS=30

# ── API Server ─────────────────────────────────────────────
VISIONNAV_API__HOST=0.0.0.0
VISIONNAV_API__PORT=8000
VISIONNAV_API__CORS_ORIGINS=["*"]
# Comma-separated valid API keys
VISIONNAV_API__VALID_KEYS=["dev-key-change-me"]

# ── Agent Settings ─────────────────────────────────────────
VISIONNAV_AGENT__MAX_STEPS=50
VISIONNAV_AGENT__SCREENSHOT_DIR=/tmp/visionnav/screenshots
VISIONNAV_AGENT__CHANGE_THRESHOLD=0.01

# ── Database ───────────────────────────────────────────────
# SQLite for dev, swap to PostgreSQL URL for production
VISIONNAV_DB__URL=sqlite+aiosqlite:///./visionnav.db

# ── External Services ──────────────────────────────────────
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=visionnav
"""

MAKEFILE = """
.PHONY: help dev test test-unit test-integration lint format train-stage1 \\
        train-stage2 train-stage3 eval serve-api serve-vllm docker-build docker-dev

help:  ## Show this help
\t@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \\
\t awk 'BEGIN {FS = ":.*##"}; {printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

# ── Development ────────────────────────────────────────────────────────────────
dev:  ## Set up development environment
\tuv sync --all-extras
\tuv run pre-commit install
\t@echo "\\n✓ Dev environment ready. Copy .env.example to .env and fill in values."

# ── Testing ────────────────────────────────────────────────────────────────────
test:  ## Run all tests with coverage
\tuv run pytest tests/unit tests/integration -v --cov=src/visionnav --cov-report=term-missing

test-unit:  ## Run unit tests only (fast, no I/O)
\tuv run pytest tests/unit -v -q

test-integration:  ## Run integration tests (requires services)
\tuv run pytest tests/integration -v -m integration

test-e2e:  ## Run end-to-end tests (sandboxed)
\tuv run pytest tests/e2e -v -m e2e

# ── Code Quality ───────────────────────────────────────────────────────────────
lint:  ## Run linter and type checker
\tuv run ruff check src/ tests/ data_pipeline/
\tuv run ruff format --check src/ tests/
\tuv run mypy src/visionnav

format:  ## Auto-fix linting issues
\tuv run ruff check --fix src/ tests/ data_pipeline/
\tuv run ruff format src/ tests/

# ── Data Pipeline ──────────────────────────────────────────────────────────────
data-download:  ## Download GUI-Net-1M dataset
\tuv run python -m data_pipeline.pipeline --stage download

data-clean:  ## Clean and normalize dataset
\tuv run python -m data_pipeline.pipeline --stage clean

data-format:  ## Convert to LLaMA-Factory format
\tuv run python -m data_pipeline.pipeline --stage format

data-all:  ## Run full data pipeline
\tuv run python -m data_pipeline.pipeline --stage all

# ── Training ───────────────────────────────────────────────────────────────────
train-stage1:  ## Stage 1: GUI grounding fine-tune
\tbash training/scripts/train_stage1.sh

train-stage2:  ## Stage 2: Action prediction fine-tune
\tbash training/scripts/train_stage2.sh

train-stage3:  ## Stage 3: Multi-step planning fine-tune
\tbash training/scripts/train_stage3.sh

eval:  ## Run benchmark evaluations
\tuv run python training/evaluation/eval_screenspot.py
\tuv run python training/evaluation/eval_mind2web.py

# ── Serving ────────────────────────────────────────────────────────────────────
serve-api:  ## Start FastAPI development server
\tuv run uvicorn visionnav.api.app:create_app --factory --reload --port 8000

serve-vllm:  ## Start vLLM inference server (requires GPU)
\tuv run vllm serve Qwen/Qwen2.5-VL-3B-Instruct \\
\t\t--port 8001 \\
\t\t--served-model-name visionnav-3b \\
\t\t--max-model-len 4096 \\
\t\t--limit-mm-per-prompt image=3

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:  ## Build all Docker images
\tdocker build -f docker/Dockerfile.api -t visionnav-api:latest .
\tdocker build -f docker/Dockerfile.inference -t visionnav-inference:latest .

docker-dev:  ## Start development Docker stack
\tdocker compose -f docker/docker-compose.dev.yml up -d

docker-down:  ## Stop development Docker stack
\tdocker compose -f docker/docker-compose.dev.yml down
"""

README = """
# VisionNav 🖥️🤖

**AI-powered GUI navigation agent** that sees your screen, understands UI elements,
and controls mouse/keyboard to complete tasks from natural language instructions.

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Model](https://img.shields.io/badge/model-Qwen2.5--VL--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## What It Does

```
"Open Chrome and search for AI news"
"Login to Gmail and find unread emails"
"Fill out this form automatically"
"Open VS Code and run the project"
```

VisionNav sees your screen → understands UI elements → reasons about the task →
executes mouse/keyboard actions → verifies results → retries on failure.

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/AdnanKhaan11/visionnav.git
cd visionnav
cp .env.example .env        # Fill in your values

# 2. Install dependencies
uv sync --all-extras        # or: pip install -e ".[dev]"

# 3. Start the API server
make serve-api

# 4. Submit a task
curl -X POST http://localhost:8000/v1/tasks/ \\
  -H "Authorization: Bearer dev-key-change-me" \\
  -H "Content-Type: application/json" \\
  -d '{"instruction": "Open calculator"}'
```

## Architecture

```
User Instruction
      ↓
  FastAPI (/v1/tasks)
      ↓
  VisionNavAgent
      ↓
  ┌───────────────────────────────────┐
  │ Perceive → Reason → Act → Verify │
  └───────────────────────────────────┘
      ↓
  PlatformAdapter (Windows/macOS/Linux/Android)
```

## Documentation

- [Training Guide](docs/training.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## Project Structure

See [MVP Architecture](docs/architecture.md) for the full folder structure explanation.

## License

MIT — see [LICENSE](LICENSE)
"""

LICENSE_MIT = """
MIT License

Copyright (c) 2026 VisionNav

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# ─── pyproject.toml ───────────────────────────────────────────────────────────

PYPROJECT_TOML = """
[project]
name = "visionnav"
version = "0.1.0"
description = "AI-powered GUI navigation agent — sees your screen, controls your computer"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }

dependencies = [
    # AI / Model inference
    "transformers>=4.51.3",
    "qwen-vl-utils>=0.0.11",
    "accelerate>=1.2.1",
    "peft>=0.15.1",
    "torch>=2.3.0",

    # vLLM HTTP client (vLLM server runs separately)
    "httpx>=0.27.0",

    # OCR
    "paddleocr>=2.7.0",
    "pytesseract>=0.3.10",

    # Screen capture + automation
    "mss>=9.0.0",
    "pyautogui>=0.9.54",
    "pynput>=1.7.6",
    "Pillow>=11.0.0",
    "numpy>=1.26.0,<2.0.0",
    "opencv-python-headless>=4.10",

    # Android
    "adbutils>=2.7.0",
    "uiautomator2>=3.2.0",

    # API
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",

    # Database
    "sqlmodel>=0.0.19",
    "aiosqlite>=0.20.0",

    # Config
    "pyyaml>=6.0.1",
    "python-dotenv>=1.0.0",

    # Utilities
    "structlog>=24.1.0",
    "tenacity>=8.3.0",
    "imagehash>=4.3.1",
]

[project.optional-dependencies]
train = [
    "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git@main",
    "liger-kernel>=0.5.8",
    "flash-attn>=2.7.4",
    "wandb>=0.19.0",
    "datasets>=2.20.0",
    "albumentations>=1.3.0",
]
serve = [
    "vllm>=0.8.4",
]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "pre-commit>=3.7.0",
    "httpx>=0.27.0",
]
data = [
    "huggingface-hub>=0.23.0",
    "selenium>=4.32.0",
    "webdriver-manager>=4.0.2",
]

[project.scripts]
visionnav-api   = "visionnav.api.app:run_server"
visionnav-agent = "visionnav.agent.agent:run_cli"
visionnav-data  = "data_pipeline.pipeline:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/visionnav"]

[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests", "data_pipeline", "training"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "N"]
ignore = ["B008"]

[tool.ruff.lint.isort]
known-first-party = ["visionnav"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
exclude = ["notebooks/", "data_pipeline/", "training/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = ["--tb=short", "--strict-markers", "-q"]
markers = [
    "unit: fast isolated tests, no external I/O",
    "integration: requires running services",
    "e2e: full task execution in sandboxed environment",
    "slow: marks slow tests",
]

[tool.coverage.run]
source = ["src/visionnav"]
omit = ["tests/*", "notebooks/*"]

[tool.coverage.report]
fail_under = 70
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "\\\\.\\\\.\\\\.",
]
"""

# ─── CONFIGS ──────────────────────────────────────────────────────────────────

CONFIG_BASE = """
# ============================================================
# VisionNav Base Configuration
# All environments inherit from this file.
# Override values in development.yaml or production.yaml
# or via environment variables (VISIONNAV_* prefix).
# ============================================================

agent:
  max_steps: 50
  screenshot_dir: "/tmp/visionnav/screenshots"
  change_threshold: 0.01        # Fraction of pixels that must change to confirm action success

model:
  backend: "local"              # "local" | "vllm"
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  dtype: "bfloat16"
  device_map: "auto"

vllm:
  base_url: "http://localhost:8001"
  model_name: "visionnav-3b"
  timeout_seconds: 30

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  valid_keys: []                # Set via env: VISIONNAV_API__VALID_KEYS

db:
  url: "sqlite+aiosqlite:///./visionnav.db"

platforms:
  desktop:
    capture:
      monitor_index: 1
    mouse:
      move_duration: 0.15
      click_delay: 0.1
      use_bezier_path: true
    keyboard:
      typing_wpm: 60
      key_delay: 0.05
  android:
    adb:
      device_serial: null
    touch:
      tap_duration: 50
      swipe_duration: 300

ocr:
  primary: "paddleocr"
  fallback: "tesseract"
  min_confidence: 0.50
  max_regions: 30

safety:
  high_risk_requires_confirmation: true
  blocked_keywords:
    - "format disk"
    - "rm -rf"
    - "delete all"
"""

CONFIG_DEV = """
# Development overrides — applied on top of base.yaml
env: development

model:
  backend: "local"

api:
  cors_origins: ["*"]

agent:
  max_steps: 20               # Shorter for faster dev iteration
"""

CONFIG_PROD = """
# Production overrides — applied on top of base.yaml
env: production

model:
  backend: "vllm"

api:
  cors_origins:
    - "https://visionnav.ai"
    - "https://app.visionnav.ai"

agent:
  max_steps: 30

db:
  # Set via env: VISIONNAV_DB__URL=postgresql+asyncpg://user:pass@host/db
  url: ""
"""

SFT_3B_YAML = """
# ── Stage 1: GUI Grounding ─────────────────────────────────────────────────
# Goal: Teach model to locate UI elements from descriptions
# Gate: ScreenSpot Acc@0.5 >= 75% before moving to Stage 2
# GPU:  Single RTX 3090 (24GB) — ~18 hours

model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj

dataset: gui_video_full,wikihow_v3,baidu_jingyan_train,showui_desktop_augmented,showui_web
template: qwen2_vl
cutoff_len: 4096
max_samples: 999999
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: checkpoints/stage1_grounding
logging_steps: 10
save_steps: 500
eval_steps: 500
plot_loss: true
overwrite_output_dir: false

per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000

report_to: wandb
run_name: visionnav-stage1-grounding
"""

SFT_3B_STAGE2 = """
# ── Stage 2: Action Prediction ─────────────────────────────────────────────
# Goal: Given screen + task → predict correct action type + coordinates
# Gate: Action type accuracy >= 85%, coordinate error <= 5% screen width
# Resumes from Stage 1 checkpoint

model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
resume_from_checkpoint: checkpoints/stage1_grounding
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj

dataset: aitw_with_thoughts,mind2web_with_thoughts,guiact_smartphone_thought,miniwob_with_thoughts
template: qwen2_vl
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: checkpoints/stage2_action
logging_steps: 10
save_steps: 500
eval_steps: 500
plot_loss: true

per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true

report_to: wandb
run_name: visionnav-stage2-action
"""

SFT_3B_STAGE3 = """
# ── Stage 3: Multi-Step Planning ───────────────────────────────────────────
# Goal: Reason through multi-step tasks with chain-of-thought
# Gate: ScreenSpot >= 79.6%, Mind2Web Step SR >= 44%
# Resumes from Stage 2 checkpoint

model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
resume_from_checkpoint: checkpoints/stage2_action
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: q_proj,k_proj,v_proj,o_proj

dataset: visionnav_multistep,visionnav_recovery,wikihow_v3,baidu_jingyan_train
template: qwen2_vl
cutoff_len: 8192
overwrite_cache: true

output_dir: checkpoints/stage3_planning
logging_steps: 10
save_steps: 500
plot_loss: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true

report_to: wandb
run_name: visionnav-stage3-planning
"""
FILE_CONTENTS = {}
# ─── SRC/VISIONNAV FILES ─────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/__init__.py"] = '''\
"""
VisionNav — AI-powered GUI navigation agent.
GitHub: https://github.com/AdnanKhaan11/visionnav
"""
__version__ = "0.1.0"
__author__ = "VisionNav Team"
'''

FILE_CONTENTS["src/visionnav/settings.py"] = '''\
"""
VisionNav typed settings.
All configuration lives here — no hardcoded values anywhere else.

Priority (highest → lowest):
  1. Environment variables   VISIONNAV_MODEL__BACKEND=vllm
  2. .env file
  3. Defaults defined below
"""
from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLLMSettings(BaseSettings):
    base_url: str = "http://localhost:8001"
    model_name: str = "visionnav-3b"
    timeout_seconds: int = 30


class ModelSettings(BaseSettings):
    backend: str = "local"
    name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    vllm: VLLMSettings = VLLMSettings()


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    valid_keys: list[str] = Field(default_factory=list)


class AgentSettings(BaseSettings):
    max_steps: int = 50
    screenshot_dir: str = "/tmp/visionnav/screenshots"
    change_threshold: float = 0.01


class DBSettings(BaseSettings):
    url: str = "sqlite+aiosqlite:///./visionnav.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VISIONNAV_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )
    env: str = "development"
    model: ModelSettings = ModelSettings()
    api: APISettings = APISettings()
    agent: AgentSettings = AgentSettings()
    db: DBSettings = DBSettings()


def get_settings() -> Settings:
    return Settings()
'''

# ─── AGENT ───────────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/agent/__init__.py"] = (
    '"""Agent loop — the core orchestrator of VisionNav."""\n'
)

FILE_CONTENTS["src/visionnav/agent/state.py"] = '''\
"""AgentState — immutable snapshot of one agent step."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from visionnav.actions.schema import Action


@dataclass(frozen=True)
class AgentState:
    step_index: int
    task_instruction: str
    screenshot_path: str
    ocr_text: str
    action_taken: Optional["Action"]
    action_success: bool
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    def to_history_entry(self) -> dict:
        action_str = ""
        if self.action_taken:
            action_str = f"{self.action_taken.type} | success={self.action_success}"
        return {"role": "assistant", "content": f"Step {self.step_index}: {action_str}"}


@dataclass
class TaskResult:
    task_id: str
    success: bool
    steps: int
    error: Optional[str] = None
    summary: str = ""
'''

FILE_CONTENTS["src/visionnav/agent/planner.py"] = '''\
"""Task Planner — decomposes a task string into ordered step descriptions."""
from __future__ import annotations
import structlog

log = structlog.get_logger(__name__)

_TASK_TEMPLATES: dict[str, list[str]] = {
    "open":   ["Find the target app or file", "Click to open it",
                "Wait for it to load", "Verify it opened successfully"],
    "search": ["Open the search interface", "Click the search field",
                "Type the search query", "Press Enter", "Wait for results"],
    "login":  ["Navigate to the login page", "Enter username",
                "Enter password", "Click login", "Verify login succeeded"],
    "fill":   ["Identify all form fields", "Fill each field in order",
                "Review the filled values", "Submit the form", "Verify submission"],
    "click":  ["Locate the target element", "Click on it", "Verify the action worked"],
    "type":   ["Find the input field", "Click to focus it",
                "Type the text", "Verify text was entered"],
    "scroll": ["Determine scroll direction", "Scroll to reveal content",
                "Verify target is visible"],
}

_GENERIC_PLAN = [
    "Analyse the current screen state",
    "Identify the action needed for the task",
    "Execute the action",
    "Verify the result",
    "Continue until task is complete",
]


class TaskPlanner:
    def decompose(self, task: str) -> list[str]:
        task_lower = task.lower()
        for keyword, steps in _TASK_TEMPLATES.items():
            if keyword in task_lower:
                log.debug("plan_matched", keyword=keyword, steps=len(steps))
                return steps
        log.debug("plan_generic", task=task[:60])
        return _GENERIC_PLAN
'''

FILE_CONTENTS["src/visionnav/agent/reporter.py"] = '''\
"""Task Reporter — human-readable summary from step history."""
from __future__ import annotations
from visionnav.agent.state import AgentState, TaskResult


class TaskReporter:
    def generate(self, result: TaskResult, history: list[AgentState]) -> str:
        status = "SUCCESS" if result.success else "FAILED"
        lines = [
            f"Task Report — {status}",
            "=" * 50,
            f"Task ID : {result.task_id}",
            f"Steps   : {result.steps}",
            f"Outcome : {result.summary or result.error or status}",
            "=" * 50,
            "Step Log:",
        ]
        for s in history:
            info = "—"
            if s.action_taken:
                info = f"{s.action_taken.type} {'OK' if s.action_success else 'FAIL'}"
            lines.append(f"  [{s.step_index:02d}] {info}")
            if s.error:
                lines.append(f"       Error: {s.error}")
        return "\n".join(lines)
'''

FILE_CONTENTS["src/visionnav/agent/agent.py"] = '''\
"""VisionNavAgent — main agent loop (perceive → reason → act → verify)."""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime
from pathlib import Path

import structlog

from visionnav.actions.executor import ActionExecutor
from visionnav.actions.parser import ActionParseError, parse_action
from visionnav.actions.schema import Action, ActionType
from visionnav.actions.verifier import ActionVerifier
from visionnav.agent.planner import TaskPlanner
from visionnav.agent.reporter import TaskReporter
from visionnav.agent.state import AgentState, TaskResult
from visionnav.memory.base import MemoryStore
from visionnav.models.base import ModelBackend
from visionnav.perception.fusion import fuse
from visionnav.perception.ocr import OCREngine
from visionnav.platforms.base import PlatformAdapter
from visionnav.safety.classifier import RiskLevel, SafetyClassifier
from visionnav.settings import AgentSettings
from visionnav.utils.image import save_screenshot
from visionnav.utils.logging import get_logger

log = get_logger(__name__)


def _extract_reasoning(text: str) -> str:
    import re
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


class VisionNavAgent:
    """
    Core agent. Inject all dependencies — nothing is constructed internally.
    Swap model/platform/memory implementations without touching this file.
    """

    def __init__(
        self,
        model: ModelBackend,
        platform: PlatformAdapter,
        memory: MemoryStore,
        safety: SafetyClassifier,
        settings: AgentSettings,
    ) -> None:
        self._model    = model
        self._platform = platform
        self._memory   = memory
        self._safety   = safety
        self._settings = settings
        self._ocr      = OCREngine()
        self._executor = ActionExecutor(platform)
        self._verifier = ActionVerifier(settings.change_threshold)
        self._planner  = TaskPlanner()
        self._reporter = TaskReporter()

    async def run(self, task_id: str, task: str) -> TaskResult:
        bound = log.bind(task_id=task_id)
        bound.info("task_started", instruction=task[:100])

        await self._memory.save_task(task_id, task)
        plan    = self._planner.decompose(task)
        history: list[AgentState] = []

        ss_dir = Path(self._settings.screenshot_dir) / task_id
        ss_dir.mkdir(parents=True, exist_ok=True)

        for step_num in range(self._settings.max_steps):
            bound.info("step_started", step=step_num)
            t0 = datetime.utcnow()

            # 1. PERCEIVE
            arr, meta       = await self._platform.capture()
            ss_path         = ss_dir / f"step_{step_num:03d}.png"
            save_screenshot(arr, ss_path)
            meta["path"]    = str(ss_path)
            ocr_regions     = self._ocr.run(arr)
            ui_elements     = await self._platform.get_ui_tree()
            observation     = fuse(arr, meta, ocr_regions, ui_elements)

            # 2. REASON
            history_dicts = [s.to_history_entry() for s in history[-10:]]
            raw_output    = await self._model.predict_action(
                observation, task, history_dicts, plan
            )
            reasoning = _extract_reasoning(raw_output)

            # 3. PARSE
            try:
                action = parse_action(raw_output)
            except ActionParseError as exc:
                bound.warning("parse_failed", step=step_num, error=str(exc))
                action = Action(
                    type=ActionType.FAIL,
                    description=f"Model output unparseable: {exc}",
                )

            bound.info("action_planned", step=step_num, action=action.type)

            # 4. SAFETY
            risk = self._safety.classify(action, context=observation.to_text_summary())
            if risk >= RiskLevel.HIGH:
                bound.warning("action_blocked", action=action.type, risk=risk.name)
                action = Action(
                    type=ActionType.FAIL,
                    description=f"High-risk action blocked ({risk.name}): {action.description}",
                )

            # 5. EXECUTE
            w, h   = self._platform.get_screen_size()
            await self._executor.execute(action, w, h)

            # 6. VERIFY
            after_arr, _ = await self._platform.capture()
            success, change = self._verifier.verify(arr, after_arr, action)

            elapsed = (datetime.utcnow() - t0).total_seconds()
            bound.info("step_complete", step=step_num, action=action.type,
                       success=success, change=round(change, 3), elapsed_s=round(elapsed, 2))

            # 7. RECORD
            state = AgentState(
                step_index=step_num,
                task_instruction=task,
                screenshot_path=str(ss_path),
                ocr_text=observation.to_text_summary()[:500],
                action_taken=action,
                action_success=success,
                reasoning=reasoning[:1000],
                error=None if success else f"No effect detected (change={change:.3f})",
            )
            await self._memory.save_step(task_id, state)
            history.append(state)

            # 8. TERMINAL
            if action.type == ActionType.DONE:
                await self._memory.mark_task_complete(task_id, True, action.description)
                bound.info("task_completed", steps=step_num + 1)
                return TaskResult(task_id=task_id, success=True,
                                  steps=step_num + 1, summary=action.description)

            if action.type == ActionType.FAIL:
                await self._memory.mark_task_complete(task_id, False, action.description)
                bound.warning("task_failed", steps=step_num + 1, reason=action.description)
                return TaskResult(task_id=task_id, success=False,
                                  steps=step_num + 1, error=action.description)

        await self._memory.mark_task_complete(task_id, False, "Max steps reached")
        bound.warning("task_max_steps", max=self._settings.max_steps)
        return TaskResult(task_id=task_id, success=False,
                          steps=self._settings.max_steps,
                          error="Maximum steps reached without completing the task.")


async def run_cli() -> None:
    """Entry point: visionnav-agent 'Open Chrome'"""
    import sys
    from visionnav.settings import Settings
    from visionnav.models.local import LocalModelBackend
    from visionnav.platforms.desktop import DesktopPlatform
    from visionnav.memory.sqlite import SQLiteMemoryStore
    from visionnav.safety.classifier import SafetyClassifier

    if len(sys.argv) < 2:
        print("Usage: visionnav-agent 'Your task here'")
        sys.exit(1)

    task     = " ".join(sys.argv[1:])
    settings = Settings()
    agent    = VisionNavAgent(
        model    = LocalModelBackend(settings.model),
        platform = DesktopPlatform(),
        memory   = SQLiteMemoryStore(settings.db.url),
        safety   = SafetyClassifier(),
        settings = settings.agent,
    )
    result = await agent.run(str(uuid.uuid4()), task)
    print("\n" + ("✅ Success" if result.success else "❌ Failed"))
    print(f"Steps: {result.steps}")
    if result.error:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(run_cli())
'''

# ─── PERCEPTION ──────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/perception/__init__.py"] = (
    '"""Perception — screen capture, OCR, UI tree, and observation fusion."""\n'
)

FILE_CONTENTS["src/visionnav/perception/capture.py"] = '''\
"""Cross-platform screenshot capture using mss."""
from __future__ import annotations
import numpy as np
import mss


class ScreenCapture:
    def __init__(self, monitor_index: int = 1) -> None:
        self._monitor_index = monitor_index

    def capture(self) -> tuple[np.ndarray, dict]:
        with mss.mss() as sct:
            mon  = sct.monitors[self._monitor_index]
            shot = sct.grab(mon)
            arr  = np.frombuffer(shot.rgb, dtype=np.uint8)
            arr  = arr.reshape(shot.height, shot.width, 3)
            meta = {"width": shot.width, "height": shot.height,
                    "monitor": self._monitor_index}
        return arr, meta

    def get_screen_size(self) -> tuple[int, int]:
        with mss.mss() as sct:
            m = sct.monitors[self._monitor_index]
            return m["width"], m["height"]
'''

FILE_CONTENTS["src/visionnav/perception/ocr.py"] = '''\
"""OCR Engine — PaddleOCR primary, Tesseract fallback."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class TextRegion:
    text: str
    bbox: tuple[float, float, float, float]   # normalised (x1,y1,x2,y2)
    confidence: float


class OCREngine:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self._min_confidence = min_confidence
        self._paddle = None   # lazy-loaded

    def run(self, image: np.ndarray) -> list[TextRegion]:
        h, w = image.shape[:2]
        try:
            return self._run_paddle(image, w, h)
        except Exception as exc:
            log.warning("paddle_failed", error=str(exc))
            return self._run_tesseract(image, w, h)

    def _run_paddle(self, image: np.ndarray, w: int, h: int) -> list[TextRegion]:
        if self._paddle is None:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        results = self._paddle.ocr(image, cls=True)
        regions: list[TextRegion] = []
        if not results or not results[0]:
            return regions
        for line in results[0]:
            pts, (text, conf) = line
            if conf < self._min_confidence:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            regions.append(TextRegion(
                text=text.strip(),
                bbox=(min(xs)/w, min(ys)/h, max(xs)/w, max(ys)/h),
                confidence=float(conf),
            ))
        return regions

    def _run_tesseract(self, image: np.ndarray, w: int, h: int) -> list[TextRegion]:
        import pytesseract
        from PIL import Image
        data = pytesseract.image_to_data(
            Image.fromarray(image), output_type=pytesseract.Output.DICT
        )
        regions: list[TextRegion] = []
        for i, text in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf < int(self._min_confidence * 100) or not text.strip():
                continue
            x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            regions.append(TextRegion(
                text=text.strip(),
                bbox=(x/w, y/h, (x+bw)/w, (y+bh)/h),
                confidence=conf / 100.0,
            ))
        return regions
'''

FILE_CONTENTS["src/visionnav/perception/ui_tree.py"] = '''\
"""OS accessibility tree extraction."""
from __future__ import annotations
import platform
import structlog

log = structlog.get_logger(__name__)


def get_ui_tree() -> list[dict]:
    """Return UI element list. Returns [] gracefully if unavailable."""
    system = platform.system()
    try:
        if system == "Windows":  return _windows()
        if system == "Darwin":   return _macos()
        if system == "Linux":    return _linux()
        return []
    except Exception as exc:
        log.warning("ui_tree_failed", system=system, error=str(exc))
        return []


def _windows() -> list[dict]:
    # TODO Phase 5: pywinauto / comtypes UIAutomation
    return []

def _macos() -> list[dict]:
    # TODO Phase 5: ApplicationServices / Quartz Accessibility API
    return []

def _linux() -> list[dict]:
    # TODO Phase 5: pyatspi AT-SPI
    return []
'''

FILE_CONTENTS["src/visionnav/perception/fusion.py"] = '''\
"""Merge screenshot + OCR + UI tree into a single Observation."""
from __future__ import annotations
import base64
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from visionnav.perception.ocr import TextRegion


@dataclass
class Observation:
    screenshot_b64: str
    screenshot_path: str
    screen_width: int
    screen_height: int
    ocr_regions: list[TextRegion] = field(default_factory=list)
    ui_elements: list[dict]       = field(default_factory=list)
    platform: str                 = "desktop"

    def to_text_summary(self, max_ocr: int = 30) -> str:
        lines: list[str] = []
        if self.ocr_regions:
            lines.append("Detected text on screen:")
            for r in self.ocr_regions[:max_ocr]:
                x1, y1, x2, y2 = r.bbox
                lines.append(f"  - \'{r.text}\' at [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]")
        if self.ui_elements:
            lines.append("\\nInteractive elements:")
            for el in self.ui_elements[:20]:
                lines.append(
                    f"  - [{el.get(\'type\',\'?\')}] \'{el.get(\'label\',\'\')}\'"
                    f" at {el.get(\'bounds\',\'?\')}"
                )
        return "\\n".join(lines) if lines else "Screen appears empty or unreadable."


def fuse(
    image: np.ndarray,
    meta: dict,
    ocr_regions: list[TextRegion],
    ui_elements: list[dict],
) -> Observation:
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return Observation(
        screenshot_b64=b64,
        screenshot_path=meta.get("path", ""),
        screen_width=meta.get("width", image.shape[1]),
        screen_height=meta.get("height", image.shape[0]),
        ocr_regions=ocr_regions,
        ui_elements=ui_elements,
        platform=meta.get("platform", "desktop"),
    )
'''

# ─── MODELS ──────────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/models/__init__.py"] = (
    '"""Model backends — swappable VLM inference implementations."""\n'
)

FILE_CONTENTS["src/visionnav/models/base.py"] = '''\
"""ModelBackend Protocol — the interface every VLM backend must satisfy."""
from __future__ import annotations
from typing import Protocol, runtime_checkable
from visionnav.perception.fusion import Observation


@runtime_checkable
class ModelBackend(Protocol):
    async def predict_action(
        self,
        observation: Observation,
        task: str,
        history: list[dict],
        plan: list[str],
    ) -> str: ...
'''

FILE_CONTENTS["src/visionnav/models/prompt.py"] = '''\
"""Prompt builder for Qwen2.5-VL."""
from __future__ import annotations
from visionnav.perception.fusion import Observation

SYSTEM_PROMPT = """You are VisionNav, an AI agent that controls computer interfaces.

Always respond with TWO blocks:

<think>
Analyse the screen. What is visible? Where is the target element?
What is the best next single action to progress the task?
</think>
<action>
{"type": "ACTION_TYPE", "coordinates": [x, y], "text": "...", "description": "..."}
</action>

ACTION_TYPES:
  click, double_click, right_click, type, key, scroll, wait, done, fail

RULES:
- coordinates are normalised floats in [0.0, 1.0]  (0,0 = top-left, 1,1 = bottom-right)
- always think before acting
- use "done" when the task is fully complete
- use "fail" only when the task is truly impossible
"""


def build_prompt(
    observation: Observation,
    task: str,
    history: list[dict],
    plan: list[str],
) -> list[dict]:
    plan_text    = "\\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
    history_text = "\\n".join(
        f"  Step {i}: {h.get(\'content\', \'\')}" for i, h in enumerate(history[-5:])
    ) or "  None yet"

    user_text = (
        f"TASK: {task}\\n\\n"
        f"PLAN:\\n{plan_text}\\n\\n"
        f"PREVIOUS ACTIONS:\\n{history_text}\\n\\n"
        f"CURRENT SCREEN (OCR):\\n{observation.to_text_summary()}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{observation.screenshot_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
'''

FILE_CONTENTS["src/visionnav/models/local.py"] = '''\
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
'''

FILE_CONTENTS["src/visionnav/models/vllm_backend.py"] = '''\
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
'''

# ─── ACTIONS ─────────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/actions/__init__.py"] = (
    '"""Actions — schema, parser, executor, verifier."""\n'
)

FILE_CONTENTS["src/visionnav/actions/schema.py"] = '''\
"""Typed Action schema."""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    CLICK        = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK  = "right_click"
    LONG_PRESS   = "long_press"
    TYPE         = "type"
    KEY          = "key"
    SCROLL       = "scroll"
    SWIPE        = "swipe"
    DRAG         = "drag"
    WAIT         = "wait"
    SCREENSHOT   = "screenshot"
    DONE         = "done"
    FAIL         = "fail"


class Action(BaseModel):
    type:        ActionType
    coordinates: Optional[tuple[float, float]] = None
    text:        Optional[str]  = None
    key:         Optional[str]  = None
    direction:   Optional[str]  = None
    amount:      int            = 0
    duration_ms: int            = 0
    description: str            = ""
    confidence:  float          = Field(default=1.0, ge=0.0, le=1.0)
'''

FILE_CONTENTS["src/visionnav/actions/parser.py"] = '''\
"""Parse raw model text → typed Action. Defensive against all malformed output."""
from __future__ import annotations
import json
import re
from visionnav.actions.schema import Action, ActionType

_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)


class ActionParseError(ValueError):
    """Raised when model output cannot be parsed into a valid Action."""


def parse_action(model_output: str) -> Action:
    match = _PATTERN.search(model_output)
    if not match:
        raise ActionParseError(f"No <action> block found. Preview: {model_output[:200]!r}")

    try:
        raw = json.loads(match.group(1).strip())
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"Invalid JSON: {exc}") from exc

    try:
        action_type = ActionType(raw.get("type", ""))
    except ValueError:
        raise ActionParseError(f"Unknown action type: {raw.get('type')!r}")

    coords = raw.get("coordinates")
    if coords is not None:
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            raise ActionParseError(f"coordinates must be [x, y], got: {coords!r}")
        nx, ny = float(coords[0]), float(coords[1])
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            raise ActionParseError(f"coordinates out of [0,1]: [{nx},{ny}]")
        coords = (nx, ny)

    return Action(
        type=action_type,
        coordinates=coords,
        text=raw.get("text"),
        key=raw.get("key"),
        direction=raw.get("direction"),
        amount=int(raw.get("amount", 0)),
        duration_ms=int(raw.get("duration_ms", 0)),
        description=str(raw.get("description", "")),
        confidence=float(raw.get("confidence", 1.0)),
    )
'''

FILE_CONTENTS["src/visionnav/actions/executor.py"] = '''\
"""Dispatch typed Actions to the platform adapter."""
from __future__ import annotations
import asyncio
import structlog
from visionnav.actions.schema import Action, ActionType
from visionnav.platforms.base import PlatformAdapter
from visionnav.utils.coords import denormalize

log = structlog.get_logger(__name__)


class ActionExecutor:
    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    async def execute(self, action: Action, screen_w: int, screen_h: int) -> bool:
        try:
            return await self._dispatch(action, screen_w, screen_h)
        except Exception as exc:
            log.error("execute_failed", action=action.type, error=str(exc))
            return False

    async def _dispatch(self, action: Action, w: int, h: int) -> bool:
        match action.type:
            case ActionType.CLICK:
                x, y = self._abs(action, w, h)
                return await self._platform.execute_click(x, y, "left")
            case ActionType.DOUBLE_CLICK:
                x, y = self._abs(action, w, h)
                return await self._platform.execute_click(x, y, "left")
            case ActionType.RIGHT_CLICK:
                x, y = self._abs(action, w, h)
                return await self._platform.execute_click(x, y, "right")
            case ActionType.TYPE:
                return await self._platform.execute_type(action.text or "")
            case ActionType.KEY:
                return await self._platform.execute_key(action.key or "")
            case ActionType.SCROLL:
                x, y = self._abs(action, w, h)
                return await self._platform.execute_scroll(
                    x, y, action.direction or "down", action.amount or 3)
            case ActionType.WAIT:
                await asyncio.sleep(action.duration_ms / 1000.0)
                return True
            case ActionType.DONE | ActionType.FAIL | ActionType.SCREENSHOT:
                return True
            case _:
                log.warning("unknown_action", type=action.type)
                return False

    def _abs(self, action: Action, w: int, h: int) -> tuple[int, int]:
        if not action.coordinates:
            raise ValueError(f"{action.type} requires coordinates")
        return denormalize(action.coordinates[0], action.coordinates[1], w, h)
'''

FILE_CONTENTS["src/visionnav/actions/verifier.py"] = '''\
"""Compare before/after screenshots to detect whether an action had effect."""
from __future__ import annotations
import numpy as np
from visionnav.actions.schema import Action, ActionType

_NO_VERIFY = {ActionType.DONE, ActionType.FAIL, ActionType.WAIT, ActionType.SCREENSHOT}


class ActionVerifier:
    def __init__(self, change_threshold: float = 0.01) -> None:
        self._threshold = change_threshold

    def verify(
        self, before: np.ndarray, after: np.ndarray, action: Action
    ) -> tuple[bool, float]:
        if action.type in _NO_VERIFY:
            return True, 0.0
        if before.shape != after.shape:
            return True, 1.0
        diff         = np.abs(before.astype(np.int16) - after.astype(np.int16))
        change_ratio = float((diff > 30).mean())
        return change_ratio >= self._threshold, change_ratio
'''

# ─── PLATFORMS ───────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/platforms/__init__.py"] = (
    '"""Platform adapters — OS-specific automation."""\n'
)

FILE_CONTENTS["src/visionnav/platforms/base.py"] = '''\
"""PlatformAdapter — abstract interface for all platforms."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class PlatformAdapter(ABC):
    @abstractmethod
    async def capture(self) -> tuple[np.ndarray, dict]: ...
    @abstractmethod
    async def get_ui_tree(self) -> list[dict]: ...
    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]: ...
    @abstractmethod
    async def execute_click(self, x: int, y: int, button: str = "left") -> bool: ...
    @abstractmethod
    async def execute_type(self, text: str) -> bool: ...
    @abstractmethod
    async def execute_scroll(self, x: int, y: int, direction: str, amount: int) -> bool: ...
    @abstractmethod
    async def execute_key(self, key_combo: str) -> bool: ...
'''

FILE_CONTENTS["src/visionnav/platforms/desktop.py"] = '''\
"""Desktop automation — Windows / macOS / Linux via pyautogui + mss."""
from __future__ import annotations
import asyncio
import numpy as np
import pyautogui
import structlog
from visionnav.perception.capture import ScreenCapture
from visionnav.perception.ui_tree import get_ui_tree
from visionnav.platforms.base import PlatformAdapter

log = structlog.get_logger(__name__)
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.05


class DesktopPlatform(PlatformAdapter):
    def __init__(self, monitor_index: int = 1) -> None:
        self._capture = ScreenCapture(monitor_index)

    async def capture(self) -> tuple[np.ndarray, dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture.capture)

    async def get_ui_tree(self) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_ui_tree)

    def get_screen_size(self) -> tuple[int, int]:
        return self._capture.get_screen_size()

    async def execute_click(self, x: int, y: int, button: str = "left") -> bool:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.click(x, y, button=button))
        return True

    async def execute_type(self, text: str) -> bool:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.write(text, interval=0.03))
        return True

    async def execute_scroll(self, x: int, y: int, direction: str, amount: int) -> bool:
        clicks = amount if direction == "up" else -amount
        loop   = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.scroll(clicks, x=x, y=y))
        return True

    async def execute_key(self, key_combo: str) -> bool:
        keys = key_combo.lower().split("+")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.hotkey(*keys))
        return True
'''

FILE_CONTENTS["src/visionnav/platforms/android.py"] = '''\
"""Android automation via ADB."""
from __future__ import annotations
import io
import numpy as np
import structlog
from visionnav.platforms.base import PlatformAdapter

log = structlog.get_logger(__name__)


class AndroidPlatform(PlatformAdapter):
    def __init__(self, serial: str | None = None) -> None:
        import adbutils
        self._device = adbutils.adb.device(serial=serial)
        log.info("android_connected", serial=self._device.serial)

    async def capture(self) -> tuple[np.ndarray, dict]:
        from PIL import Image
        img  = Image.open(io.BytesIO(self._device.screencap())).convert("RGB")
        arr  = np.array(img)
        meta = {"width": img.width, "height": img.height, "platform": "android"}
        return arr, meta

    async def get_ui_tree(self) -> list[dict]:
        return []   # TODO Phase 5: UIAutomator2 dump

    def get_screen_size(self) -> tuple[int, int]:
        info = self._device.window_size()
        return info.width, info.height

    async def execute_click(self, x: int, y: int, button: str = "left") -> bool:
        self._device.shell(f"input tap {x} {y}")
        return True

    async def execute_type(self, text: str) -> bool:
        escaped = text.replace(" ", "%s").replace("'", "\\\\'")
        self._device.shell(f"input text '{escaped}'")
        return True

    async def execute_scroll(self, x: int, y: int, direction: str, amount: int) -> bool:
        dist = amount * 300
        if direction == "down":
            self._device.shell(f"input swipe {x} {y} {x} {y-dist} 300")
        else:
            self._device.shell(f"input swipe {x} {y} {x} {y+dist} 300")
        return True

    async def execute_key(self, key_combo: str) -> bool:
        kmap = {"enter":"66","back":"4","home":"3","tab":"61"}
        kc   = kmap.get(key_combo.lower(), "")
        if kc:
            self._device.shell(f"input keyevent {kc}")
        return bool(kc)
'''

# ─── MEMORY ──────────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/memory/__init__.py"] = (
    '"""Memory — agent state persistence (swappable backends)."""\n'
)

FILE_CONTENTS["src/visionnav/memory/base.py"] = '''\
"""MemoryStore Protocol — swap SQLite → PostgreSQL by changing one file."""
from __future__ import annotations
from typing import Protocol
from visionnav.agent.state import AgentState


class MemoryStore(Protocol):
    async def save_task(self, task_id: str, instruction: str) -> None: ...
    async def save_step(self, task_id: str, state: AgentState) -> None: ...
    async def get_task_history(self, task_id: str) -> list[AgentState]: ...
    async def get_recent_steps(self, task_id: str, n: int = 10) -> list[AgentState]: ...
    async def mark_task_complete(self, task_id: str, success: bool, result: str) -> None: ...
'''

FILE_CONTENTS["src/visionnav/memory/sqlite.py"] = '''\
"""SQLite memory store — MVP. Replace with postgres.py for production."""
from __future__ import annotations
import json
import aiosqlite
import structlog
from visionnav.agent.state import AgentState

log = structlog.get_logger(__name__)


class SQLiteMemoryStore:
    def __init__(self, db_url: str = "sqlite+aiosqlite:///./visionnav.db") -> None:
        self._path = db_url.replace("sqlite+aiosqlite:///", "")

    async def _conn(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self._path)
        await conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, instruction TEXT,
            status TEXT DEFAULT 'running', result TEXT,
            created_at TEXT DEFAULT (datetime('now')))""")
        await conn.execute("""CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT, step_index INTEGER, data TEXT,
            created_at TEXT DEFAULT (datetime('now')))""")
        await conn.commit()
        return conn

    async def save_task(self, task_id: str, instruction: str) -> None:
        async with await self._conn() as c:
            await c.execute(
                "INSERT OR IGNORE INTO tasks(task_id,instruction) VALUES(?,?)",
                (task_id, instruction))
            await c.commit()

    async def save_step(self, task_id: str, state: AgentState) -> None:
        data = {
            "step_index": state.step_index, "task_instruction": state.task_instruction,
            "screenshot_path": state.screenshot_path, "ocr_text": state.ocr_text,
            "action_type": state.action_taken.type if state.action_taken else None,
            "action_success": state.action_success, "reasoning": state.reasoning,
            "error": state.error, "timestamp": state.timestamp.isoformat(),
        }
        async with await self._conn() as c:
            await c.execute(
                "INSERT INTO steps(task_id,step_index,data) VALUES(?,?,?)",
                (task_id, state.step_index, json.dumps(data)))
            await c.commit()

    async def get_task_history(self, task_id: str) -> list[AgentState]:
        async with await self._conn() as c:
            async with c.execute(
                "SELECT data FROM steps WHERE task_id=? ORDER BY step_index", (task_id,)
            ) as cur:
                rows = await cur.fetchall()
        return [_to_state(r[0]) for r in rows]

    async def get_recent_steps(self, task_id: str, n: int = 10) -> list[AgentState]:
        return (await self.get_task_history(task_id))[-n:]

    async def mark_task_complete(self, task_id: str, success: bool, result: str) -> None:
        status = "completed" if success else "failed"
        async with await self._conn() as c:
            await c.execute(
                "UPDATE tasks SET status=?,result=? WHERE task_id=?", (status, result, task_id))
            await c.commit()


def _to_state(data_json: str) -> AgentState:
    from datetime import datetime
    d = json.loads(data_json)
    return AgentState(
        step_index=d["step_index"], task_instruction=d["task_instruction"],
        screenshot_path=d["screenshot_path"], ocr_text=d["ocr_text"],
        action_taken=None, action_success=d["action_success"],
        reasoning=d["reasoning"], timestamp=datetime.fromisoformat(d["timestamp"]),
        error=d.get("error"),
    )
'''

# ─── SAFETY ──────────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/safety/__init__.py"] = (
    '"""Safety — action risk classification and execution gates."""\n'
)

FILE_CONTENTS["src/visionnav/safety/classifier.py"] = '''\
"""Risk classifier — assign a RiskLevel to each action before execution."""
from __future__ import annotations
from enum import IntEnum
from visionnav.actions.schema import Action, ActionType


class RiskLevel(IntEnum):
    SAFE    = 0
    LOW     = 1
    MEDIUM  = 2
    HIGH    = 3
    BLOCKED = 4


_BASE: dict[ActionType, RiskLevel] = {
    ActionType.SCREENSHOT: RiskLevel.SAFE,
    ActionType.WAIT:       RiskLevel.SAFE,
    ActionType.SCROLL:     RiskLevel.SAFE,
    ActionType.DONE:       RiskLevel.SAFE,
    ActionType.FAIL:       RiskLevel.SAFE,
    ActionType.CLICK:      RiskLevel.LOW,
    ActionType.DOUBLE_CLICK: RiskLevel.LOW,
    ActionType.RIGHT_CLICK:  RiskLevel.LOW,
    ActionType.KEY:        RiskLevel.LOW,
    ActionType.DRAG:       RiskLevel.LOW,
    ActionType.SWIPE:      RiskLevel.LOW,
    ActionType.LONG_PRESS: RiskLevel.LOW,
    ActionType.TYPE:       RiskLevel.MEDIUM,
}
_HIGH_CLICK = {"delete","remove","buy","purchase","send","pay","confirm","checkout","uninstall"}
_HIGH_TYPE  = {"password","card","cvv","ssn","secret","token"}


class SafetyClassifier:
    def classify(self, action: Action, context: str = "") -> RiskLevel:
        base = _BASE.get(action.type, RiskLevel.MEDIUM)
        ctx  = (action.description + " " + context).lower()
        if action.type == ActionType.CLICK and any(k in ctx for k in _HIGH_CLICK):
            return RiskLevel.HIGH
        if action.type == ActionType.TYPE and any(k in ctx for k in _HIGH_TYPE):
            return RiskLevel.HIGH
        return base
'''

FILE_CONTENTS["src/visionnav/safety/gates.py"] = '''\
"""Safety gate helpers."""
from __future__ import annotations
from visionnav.safety.classifier import RiskLevel


def should_block(risk: RiskLevel) -> bool:
    return risk >= RiskLevel.BLOCKED

def should_confirm(risk: RiskLevel, confirmation_required: bool = True) -> bool:
    return confirmation_required and risk >= RiskLevel.HIGH
'''

# ─── API ─────────────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/api/__init__.py"] = (
    '"""FastAPI application — REST interface for VisionNav."""\n'
)

FILE_CONTENTS["src/visionnav/api/app.py"] = '''\
"""FastAPI application factory."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from visionnav.api.middleware import ErrorHandlerMiddleware, RequestIDMiddleware
from visionnav.api.v1.router import v1_router
from visionnav.settings import Settings
from visionnav.utils.logging import setup_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()
    setup_logging()
    app = FastAPI(
        title="VisionNav API", version="1.0.0",
        docs_url="/docs" if settings.env != "production" else None,
        redoc_url=None,
    )
    app.add_middleware(CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    app.include_router(v1_router, prefix="/v1")
    return app


def run_server() -> None:
    import uvicorn
    s = Settings()
    uvicorn.run("visionnav.api.app:create_app", factory=True,
                host=s.api.host, port=s.api.port,
                reload=s.env == "development")
'''

FILE_CONTENTS["src/visionnav/api/dependencies.py"] = '''\
"""FastAPI dependency injection."""
from __future__ import annotations
from functools import lru_cache
from fastapi import Header, HTTPException, Depends
from visionnav.settings import Settings


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    return Settings()


async def verify_api_key(
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_cached_settings),
) -> str:
    if not settings.api.valid_keys:
        return "dev-open"
    token = authorization.replace("Bearer ", "").strip()
    if token not in settings.api.valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return token


async def get_memory():
    from visionnav.memory.sqlite import SQLiteMemoryStore
    return SQLiteMemoryStore(get_cached_settings().db.url)
'''

FILE_CONTENTS["src/visionnav/api/middleware.py"] = '''\
"""Custom middleware — request IDs and error handling."""
from __future__ import annotations
import uuid
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid      = str(uuid.uuid4())[:8]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            log.error("unhandled", path=request.url.path, error=str(exc))
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=500,
                content={"error": "internal_server_error", "detail": str(exc)})
'''

FILE_CONTENTS["src/visionnav/api/v1/__init__.py"] = '"""API v1."""\n'

FILE_CONTENTS["src/visionnav/api/v1/router.py"] = '''\
"""Aggregate all v1 routers."""
from fastapi import APIRouter
from visionnav.api.v1 import health, sessions, tasks

v1_router = APIRouter()
v1_router.include_router(health.router)
v1_router.include_router(tasks.router)
v1_router.include_router(sessions.router)
'''

FILE_CONTENTS["src/visionnav/api/v1/health.py"] = '''\
"""Health check endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"

@router.get("/health",  response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")

@router.get("/ready",   response_model=HealthResponse)
async def ready() -> HealthResponse:
    return HealthResponse(status="ok")
'''

FILE_CONTENTS["src/visionnav/api/v1/tasks.py"] = '''\
"""Task management endpoints."""
from __future__ import annotations
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from visionnav.api.dependencies import get_memory, verify_api_key
from visionnav.settings import Settings

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskRequest(BaseModel):
    instruction: str
    platform:    str = "desktop"
    max_steps:   int = 50


class TaskResponse(BaseModel):
    task_id: str
    status:  str
    message: str


@router.post("/", response_model=TaskResponse, status_code=202)
async def submit_task(
    body: TaskRequest,
    background_tasks: BackgroundTasks,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> TaskResponse:
    task_id = str(uuid.uuid4())

    async def run_task() -> None:
        from visionnav.agent.agent import VisionNavAgent
        from visionnav.memory.sqlite import SQLiteMemoryStore
        from visionnav.models.local import LocalModelBackend
        from visionnav.platforms.desktop import DesktopPlatform
        from visionnav.safety.classifier import SafetyClassifier
        s     = Settings()
        agent = VisionNavAgent(
            model=LocalModelBackend(s.model), platform=DesktopPlatform(),
            memory=SQLiteMemoryStore(s.db.url), safety=SafetyClassifier(),
            settings=s.agent,
        )
        await agent.run(task_id, body.instruction)

    background_tasks.add_task(run_task)
    return TaskResponse(task_id=task_id, status="accepted",
        message=f"Poll GET /v1/tasks/{task_id} for status.")


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> dict:
    steps = await memory.get_task_history(task_id)
    if not steps:
        raise HTTPException(404, detail=f"Task {task_id!r} not found")
    return {
        "task_id":     task_id,
        "total_steps": len(steps),
        "steps": [
            {"index": s.step_index,
             "action": s.action_taken.type if s.action_taken else None,
             "success": s.action_success, "error": s.error}
            for s in steps
        ],
    }
'''

FILE_CONTENTS["src/visionnav/api/v1/sessions.py"] = '''\
"""Session management — stub for Phase 7+."""
from fastapi import APIRouter
import uuid

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("/")
async def create_session() -> dict:
    return {"session_id": str(uuid.uuid4()), "status": "created"}
'''

# ─── UTILS ───────────────────────────────────────────────────────────────────

FILE_CONTENTS["src/visionnav/utils/__init__.py"] = (
    '"""Utilities — image, coords, retry, logging."""\n'
)

FILE_CONTENTS["src/visionnav/utils/image.py"] = '''\
"""Image helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def save_screenshot(image: np.ndarray, path: Path | str) -> None:
    from PIL import Image
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(p, "PNG")


def encode_base64(image: np.ndarray) -> str:
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
'''

FILE_CONTENTS["src/visionnav/utils/coords.py"] = '''\
"""Coordinate normalisation / denormalisation."""
from __future__ import annotations


def normalize(x: int, y: int, w: int, h: int) -> tuple[float, float]:
    return round(x / w, 4), round(y / h, 4)

def denormalize(nx: float, ny: float, w: int, h: int) -> tuple[int, int]:
    return int(nx * w), int(ny * h)

def validate_normalized(nx: float, ny: float) -> bool:
    return 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0

def bbox_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return (x1 + x2) / 2, (y1 + y2) / 2
'''

FILE_CONTENTS["src/visionnav/utils/retry.py"] = '''\
"""Retry decorator with exponential backoff."""
from __future__ import annotations
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)


def with_retry(max_attempts: int = 3, min_wait: float = 1.0,
               max_wait: float = 8.0, exceptions: tuple = (Exception,)):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )
'''

FILE_CONTENTS["src/visionnav/utils/logging.py"] = '''\
"""Structured logging setup."""
from __future__ import annotations
import logging, os
import structlog


def setup_logging(level: str | None = None) -> None:
    lvl     = level or os.getenv("LOG_LEVEL", "INFO")
    is_dev  = os.getenv("VISIONNAV_ENV", "development") == "development"
    logging.basicConfig(format="%(message)s",
                        level=getattr(logging, lvl.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if is_dev else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, lvl.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str = "") -> structlog.BoundLogger:
    return structlog.get_logger(name)
'''

# ─── TESTS ───────────────────────────────────────────────────────────────────

FILE_CONTENTS["tests/conftest.py"] = '''\
"""Shared pytest fixtures."""
from __future__ import annotations
import numpy as np
import pytest
from visionnav.actions.schema import Action, ActionType


@pytest.fixture
def blank_screen() -> np.ndarray:
    return np.zeros((600, 800, 3), dtype=np.uint8)

@pytest.fixture
def changed_screen() -> np.ndarray:
    return np.full((600, 800, 3), 255, dtype=np.uint8)

@pytest.fixture
def test_observation(blank_screen):
    import base64, io
    from PIL import Image
    from visionnav.perception.fusion import Observation
    buf = io.BytesIO()
    Image.fromarray(blank_screen).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return Observation(screenshot_b64=b64, screenshot_path="/tmp/test.png",
                       screen_width=800, screen_height=600, platform="linux")

@pytest.fixture
def click_action() -> Action:
    return Action(type=ActionType.CLICK, coordinates=(0.5, 0.5), description="test")

@pytest.fixture
def done_action() -> Action:
    return Action(type=ActionType.DONE, description="Task complete")

@pytest.fixture
def mock_model(mocker):
    m = mocker.AsyncMock()
    m.predict_action.return_value = (
        "<think>Done.</think>"
        \'<action>{"type":"done","description":"Done"}</action>\'
    )
    return m

@pytest.fixture
def mock_platform(mocker, blank_screen):
    p = mocker.AsyncMock()
    p.capture.return_value       = (blank_screen, {"width": 800, "height": 600})
    p.get_ui_tree.return_value   = []
    p.get_screen_size.return_value = (800, 600)
    p.execute_click.return_value = True
    p.execute_type.return_value  = True
    p.execute_scroll.return_value = True
    p.execute_key.return_value   = True
    return p
'''

FILE_CONTENTS["tests/unit/test_action_parser.py"] = '''\
"""Unit tests — action parser."""
import pytest
from visionnav.actions.parser import ActionParseError, parse_action
from visionnav.actions.schema import ActionType


def test_parse_click():
    out = \'<action>{"type":"click","coordinates":[0.5,0.3],"description":"OK"}</action>\'
    a   = parse_action(out)
    assert a.type == ActionType.CLICK
    assert a.coordinates == (0.5, 0.3)

def test_parse_type():
    a = parse_action(\'<action>{"type":"type","text":"hello"}</action>\')
    assert a.type == ActionType.TYPE
    assert a.text == "hello"

def test_parse_done():
    a = parse_action(\'<action>{"type":"done","description":"fin"}</action>\')
    assert a.type == ActionType.DONE

def test_missing_action_block():
    with pytest.raises(ActionParseError, match="No <action>"):
        parse_action("just text, no block")

def test_invalid_json():
    with pytest.raises(ActionParseError, match="Invalid JSON"):
        parse_action("<action>not json</action>")

def test_unknown_type():
    with pytest.raises(ActionParseError, match="Unknown action type"):
        parse_action(\'<action>{"type":"fly"}</action>\')

def test_coords_out_of_range():
    with pytest.raises(ActionParseError, match="out of"):
        parse_action(\'<action>{"type":"click","coordinates":[1.5,0.5]}</action>\')
'''

FILE_CONTENTS["tests/unit/test_coord_utils.py"] = '''\
"""Unit tests — coordinate utilities."""
from visionnav.utils.coords import normalize, denormalize, validate_normalized, bbox_center


def test_normalize_center():
    assert normalize(640, 360, 1280, 720) == (0.5, 0.5)

def test_normalize_origin():
    assert normalize(0, 0, 1280, 720) == (0.0, 0.0)

def test_round_trip():
    x, y = 320, 240
    nx, ny = normalize(x, y, 1280, 720)
    assert denormalize(nx, ny, 1280, 720) == (x, y)

def test_validate_valid():
    assert validate_normalized(0.0, 0.0) is True
    assert validate_normalized(1.0, 1.0) is True

def test_validate_invalid():
    assert validate_normalized(1.1, 0.5) is False
    assert validate_normalized(-0.1, 0.5) is False

def test_bbox_center():
    cx, cy = bbox_center(0.1, 0.1, 0.5, 0.5)
    assert cx == 0.3 and cy == 0.3
'''

FILE_CONTENTS["tests/unit/test_action_verifier.py"] = '''\
"""Unit tests — action verifier."""
import numpy as np
import pytest
from visionnav.actions.schema import Action, ActionType
from visionnav.actions.verifier import ActionVerifier


@pytest.fixture
def verifier():
    return ActionVerifier(change_threshold=0.01)

def test_detects_change(verifier):
    before = np.zeros((100,100,3), dtype=np.uint8)
    after  = np.full((100,100,3), 255, dtype=np.uint8)
    ok, r  = verifier.verify(before, after, Action(type=ActionType.CLICK, coordinates=(0.5,0.5)))
    assert ok is True and r > 0.5

def test_detects_no_change(verifier):
    s     = np.zeros((100,100,3), dtype=np.uint8)
    ok, r = verifier.verify(s, s.copy(), Action(type=ActionType.CLICK, coordinates=(0.5,0.5)))
    assert ok is False and r == 0.0

def test_done_always_ok(verifier):
    s     = np.zeros((100,100,3), dtype=np.uint8)
    ok, _ = verifier.verify(s, s.copy(), Action(type=ActionType.DONE))
    assert ok is True
'''

FILE_CONTENTS["tests/unit/test_safety_classifier.py"] = '''\
"""Unit tests — safety classifier."""
from visionnav.actions.schema import Action, ActionType
from visionnav.safety.classifier import RiskLevel, SafetyClassifier


def test_scroll_safe():
    assert SafetyClassifier().classify(
        Action(type=ActionType.SCROLL, direction="down")) == RiskLevel.SAFE

def test_click_low():
    assert SafetyClassifier().classify(
        Action(type=ActionType.CLICK, coordinates=(0.5,0.5))) == RiskLevel.LOW

def test_delete_click_high():
    assert SafetyClassifier().classify(
        Action(type=ActionType.CLICK, coordinates=(0.5,0.5), description="click delete")) == RiskLevel.HIGH

def test_password_type_high():
    assert SafetyClassifier().classify(
        Action(type=ActionType.TYPE, text="x"), context="password field") == RiskLevel.HIGH

def test_done_safe():
    assert SafetyClassifier().classify(
        Action(type=ActionType.DONE)) == RiskLevel.SAFE
'''

FILE_CONTENTS["tests/unit/test_prompt_builder.py"] = '''\
"""Unit tests — prompt builder."""
from visionnav.models.prompt import build_prompt


def test_has_system(test_observation):
    msgs = build_prompt(test_observation, "task", [], [])
    assert msgs[0]["role"] == "system"
    assert "<think>" in msgs[0]["content"]

def test_has_image(test_observation):
    msgs  = build_prompt(test_observation, "task", [], [])
    parts = msgs[1]["content"]
    imgs  = [p for p in parts if p.get("type") == "image_url"]
    assert len(imgs) == 1
    assert imgs[0]["image_url"]["url"].startswith("data:image/png;base64,")

def test_task_in_text(test_observation):
    msgs  = build_prompt(test_observation, "Open Chrome", [], [])
    texts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
    assert any("Open Chrome" in t for t in texts)
'''

FILE_CONTENTS["tests/integration/test_agent_loop.py"] = '''\
"""Integration tests — agent loop."""
from __future__ import annotations
import pytest
from visionnav.agent.agent import VisionNavAgent
from visionnav.agent.state import TaskResult
from visionnav.safety.classifier import SafetyClassifier
from visionnav.settings import AgentSettings


def _make(mock_model, mock_platform, tmp_path, mocker):
    from visionnav.memory.sqlite import SQLiteMemoryStore
    return VisionNavAgent(
        model=mock_model, platform=mock_platform,
        memory=SQLiteMemoryStore(f"sqlite+aiosqlite:///{tmp_path}/t.db"),
        safety=SafetyClassifier(),
        settings=AgentSettings(max_steps=10,
                               screenshot_dir=str(tmp_path / "ss")),
    )

@pytest.mark.integration
async def test_done_on_first_step(mock_model, mock_platform, tmp_path, mocker):
    agent  = _make(mock_model, mock_platform, tmp_path, mocker)
    result = await agent.run("t1", "Open calculator")
    assert result.success is True and result.steps == 1

@pytest.mark.integration
async def test_fail_action(mock_model, mock_platform, tmp_path, mocker):
    mock_model.predict_action.return_value = (
        "<think>nope</think>"
        \'<action>{"type":"fail","description":"impossible"}</action>\'
    )
    result = await _make(mock_model, mock_platform, tmp_path, mocker).run("t2", "x")
    assert result.success is False

@pytest.mark.integration
async def test_max_steps(mock_model, mock_platform, tmp_path, mocker):
    mock_model.predict_action.return_value = (
        "<think>clicking</think>"
        \'<action>{"type":"click","coordinates":[0.5,0.5]}</action>\'
    )
    result = await _make(mock_model, mock_platform, tmp_path, mocker).run("t3", "forever")
    assert result.steps == 10 and result.success is False
'''

FILE_CONTENTS["tests/integration/test_api_tasks.py"] = '''\
"""Integration tests — FastAPI task endpoints."""
from fastapi.testclient import TestClient
from visionnav.api.app import create_app
from visionnav.settings import APISettings, Settings


def _client():
    return TestClient(create_app(Settings(env="test",
                                          api=APISettings(valid_keys=["test-key"]))))

def test_health():
    assert _client().get("/v1/health").status_code == 200

def test_no_auth_401():
    assert _client().post("/v1/tasks/", json={"instruction": "x"}).status_code == 401

def test_submit_202():
    r = _client().post("/v1/tasks/", json={"instruction": "Open Chrome"},
                       headers={"authorization": "Bearer test-key"})
    assert r.status_code == 202
    assert "task_id" in r.json()

def test_unknown_task_404():
    r = _client().get("/v1/tasks/none",
                      headers={"authorization": "Bearer test-key"})
    assert r.status_code == 404
'''

# ─── EMPTY DIRS + RUNNER ─────────────────────────────────────────────────────

EMPTY_DIRS = [
    "checkpoints",
    "models",
    "data/raw",
    "data/processed/screenshots",
    "data/processed/annotations",
    "data/processed/ocr_cache",
    "data/augmented",
    "data/splits",
    "data/instruction_tuning",
    "data/external",
    "data/visionnav",
    "data/tongui",
    "tests/e2e",
    "scripts",
    "notebooks",
]

# ─── DATA PIPELINE FILES ─────────────────────────────────────────────────────

FILE_CONTENTS["data_pipeline/__init__.py"] = (
    '"""VisionNav data pipeline — download, clean, enrich, format training data."""\n'
)

FILE_CONTENTS["data_pipeline/pipeline.py"] = '''\
"""
VisionNav Data Pipeline Master Orchestrator
============================================
Usage:
    python -m data_pipeline.pipeline --stage all
    python -m data_pipeline.pipeline --stage download
    python -m data_pipeline.pipeline --stage clean
    python -m data_pipeline.pipeline --stage ocr
    python -m data_pipeline.pipeline --stage format
    python -m data_pipeline.pipeline --stage validate
"""
from __future__ import annotations
import argparse, sys, time


def run_download() -> None:
    from data_pipeline.downloaders import download_all
    download_all()

def run_clean() -> None:
    from data_pipeline.cleaners import run_cleaning
    run_cleaning()

def run_ocr() -> None:
    from data_pipeline.ocr_enricher import run_ocr_enrichment
    run_ocr_enrichment()

def run_format() -> None:
    from data_pipeline.formatters import run_formatting
    run_formatting()

def run_validate() -> None:
    from data_pipeline.validators import run_validation
    run_validation()


STAGES: dict[str, callable] = {
    "download": run_download,
    "clean":    run_clean,
    "ocr":      run_ocr,
    "format":   run_format,
    "validate": run_validate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionNav data pipeline")
    parser.add_argument("--stage", choices=[*STAGES.keys(), "all"], default="all")
    args   = parser.parse_args()

    stages = list(STAGES.items()) if args.stage == "all" \
             else [(args.stage, STAGES[args.stage])]

    print(f"\\n{'='*60}")
    print(f"VisionNav Data Pipeline — {len(stages)} stage(s)")
    print(f"{'='*60}\\n")

    t0 = time.time()
    for name, fn in stages:
        print(f"[{name.upper()}] Starting...")
        t = time.time()
        try:
            fn()
            print(f"[{name.upper()}] Done in {time.time()-t:.1f}s\\n")
        except Exception as exc:
            print(f"[{name.upper()}] FAILED: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"All stages complete in {time.time()-t0:.1f}s\\n")


if __name__ == "__main__":
    main()
'''

FILE_CONTENTS["data_pipeline/downloaders.py"] = '''\
"""Download GUI-Net-1M and supplementary datasets."""
from __future__ import annotations
import subprocess
from pathlib import Path

RAW_DIR = Path("data/raw")


def download_all() -> None:
    download_gui_net_1m()


def download_gui_net_1m() -> None:
    out = RAW_DIR / "gui_net_1m"
    if out.exists():
        print(f"  GUI-Net-1M already at {out} — skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    print("  Downloading GUI-Net-1M from HuggingFace...")
    subprocess.run([
        "huggingface-cli", "download",
        "Bofeee5675/GUI-Net-1M",
        "--repo-type", "dataset",
        "--local-dir", str(out),
    ], check=True)
    print(f"  GUI-Net-1M saved to {out}")
'''

FILE_CONTENTS["data_pipeline/cleaners.py"] = '''\
"""Dataset cleaning — deduplication, resolution normalisation, corrupt removal."""
from __future__ import annotations
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def run_cleaning() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("  Cleaning: dedup → normalise resolution → filter corrupt")
    # TODO Phase 1: imagehash perceptual dedup (threshold hamming < 8)
    # TODO Phase 1: resize all to 1280x720 (desktop) / 1080x1920 (mobile)
    # TODO Phase 1: remove files < 50 KB or blank screens
    print("  (stub) Cleaning complete")
'''

FILE_CONTENTS["data_pipeline/ocr_enricher.py"] = '''\
"""Pre-compute OCR for all training screenshots and cache results as JSON."""
from __future__ import annotations
from pathlib import Path

OCR_CACHE = Path("data/processed/ocr_cache")


def run_ocr_enrichment() -> None:
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    print("  Running PaddleOCR on processed screenshots...")
    # TODO Phase 2: batch PaddleOCR over all images
    # TODO Phase 2: save list[TextRegion] as JSON per image
    print("  (stub) OCR enrichment complete")
'''

FILE_CONTENTS["data_pipeline/formatters.py"] = '''\
"""Convert annotations to LLaMA-Factory sharegpt conversation format."""
from __future__ import annotations
from pathlib import Path

OUTPUT_DIR = Path("data/instruction_tuning")

SYSTEM_PROMPT = (
    "You are VisionNav, an AI GUI agent. "
    "Think inside <think>...</think> then output your action inside <action>...</action>."
)


def run_formatting() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("  Formatting samples → LLaMA-Factory sharegpt JSON...")
    # TODO Phase 1: read processed annotations
    # TODO Phase 1: add _with_thoughts chain-of-thought (TongUI approach via GPT-4o)
    # TODO Phase 1: write stage1 / stage2 / stage3 JSONL files
    print("  (stub) Formatting complete")


def make_sharegpt_sample(
    image_path: str,
    task: str,
    action_json: str,
    reasoning: str = "",
) -> dict:
    """One training sample in LLaMA-Factory sharegpt format."""
    content = ""
    if reasoning:
        content += f"<think>\\n{reasoning}\\n</think>\\n"
    content += f"<action>\\n{action_json}\\n</action>"
    return {
        "conversations": [
            {"role": "system",    "value": SYSTEM_PROMPT},
            {"role": "user",      "value": f"<image>\\nTask: {task}\\nNext action?"},
            {"role": "assistant", "value": content},
        ],
        "images": [image_path],
    }
'''

FILE_CONTENTS["data_pipeline/validators.py"] = '''\
"""Validate final dataset — schema, coordinates, completeness."""
from __future__ import annotations
import json
from pathlib import Path


def run_validation() -> None:
    files = list(Path("data/instruction_tuning").glob("*.jsonl"))
    if not files:
        print("  No files found — run --stage format first")
        return

    total = errors = 0
    for f in files:
        with open(f) as fp:
            for line in fp:
                total += 1
                try:
                    _validate(json.loads(line))
                except (json.JSONDecodeError, AssertionError) as exc:
                    errors += 1
                    print(f"  ERR {f.name}:{total} — {exc}")

    print(f"  Validated {total} samples — {errors} errors")
    if errors:
        raise SystemExit(f"Validation failed: {errors} invalid samples")
    print("  All samples valid")


def _validate(s: dict) -> None:
    assert "conversations" in s, "Missing conversations"
    assert "images" in s,        "Missing images"
    assert len(s["conversations"]) >= 2, "Need >= 2 turns"
    assert len(s["images"]) >= 1,        "Need >= 1 image"
'''

# ─── TRAINING FILES ──────────────────────────────────────────────────────────

FILE_CONTENTS["training/evaluation/eval_screenspot.py"] = '''\
"""
ScreenSpot Benchmark Evaluation
=================================
Gate: must pass threshold before promoting model to next training stage.

Usage:
    python training/evaluation/eval_screenspot.py --model checkpoints/stage1_best
    python training/evaluation/eval_screenspot.py --model checkpoints/stage1_best --threshold 75.0
"""
from __future__ import annotations
import argparse

THRESHOLDS = {
    "stage1": 75.0,   # after grounding training
    "stage2": 79.6,   # TongUI-3B baseline
    "stage3": 83.6,   # TongUI-3B-1M level
    "7b":     86.0,   # after 7B fine-tune
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--threshold", type=float, default=75.0)
    parser.add_argument("--split",     default="test", choices=["val", "test"])
    args = parser.parse_args()

    print(f"\\nScreenSpot Evaluation")
    print(f"Model     : {args.model}")
    print(f"Threshold : {args.threshold}%")
    print(f"{'='*50}")
    # TODO Phase 3: implement evaluation loop
    # Reference: TongUI-agent/tongui/eval/run_screenspot.py
    print("(stub) — implement after Stage 1 training is complete")


if __name__ == "__main__":
    main()
'''

FILE_CONTENTS["training/evaluation/eval_mind2web.py"] = '''\
"""
Mind2Web Benchmark Evaluation
================================
Metrics: Element Acc, Operation F1, Step Success Rate.

Usage:
    python training/evaluation/eval_mind2web.py --model checkpoints/stage3_best
"""
from __future__ import annotations
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="cross_task",
                        choices=["cross_task", "cross_website", "cross_domain"])
    args = parser.parse_args()
    print(f"\\nMind2Web Evaluation — {args.split}")
    print(f"Model: {args.model}")
    print("(stub) — implement after Stage 3 training")


if __name__ == "__main__":
    main()
'''

FILE_CONTENTS["training/evaluation/eval_visionnav.py"] = '''\
"""
VisionNav Internal Benchmark — 100 handcrafted tasks.
All tasks run in isolated sandboxed environments.

Usage:
    python training/evaluation/eval_visionnav.py --model checkpoints/stage3_best
"""
from __future__ import annotations
import argparse

TASKS = [
    {"id": "br_001", "task": "Open Chrome and search for AI news",  "platform": "desktop"},
    {"id": "br_002", "task": "Navigate to github.com",              "platform": "desktop"},
    {"id": "em_001", "task": "Open Gmail and find unread emails",   "platform": "desktop"},
    {"id": "sy_001", "task": "Open system settings",                "platform": "desktop"},
    {"id": "fm_001", "task": "Fill a simple contact form",          "platform": "desktop"},
    {"id": "an_001", "task": "Open Settings on Android",            "platform": "android"},
    # Add more tasks as the project grows
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--platform", default="all")
    args  = parser.parse_args()
    tasks = [t for t in TASKS
             if args.platform == "all" or t["platform"] == args.platform]
    print(f"\\nVisionNav Benchmark — {len(tasks)} tasks | model: {args.model}")
    print("(stub) — implement after Phase 6 agent loop is complete")


if __name__ == "__main__":
    main()
'''

FILE_CONTENTS["training/scripts/prepare_data.sh"] = """\
#!/usr/bin/env bash
# Run full data preparation pipeline
set -euo pipefail
echo "========================================"
echo "VisionNav — Data Preparation Pipeline"
echo "========================================"
uv run python -m data_pipeline.pipeline --stage all
echo ""
echo "Data ready. Next: make train-stage1"
"""

FILE_CONTENTS["training/scripts/train_stage1.sh"] = """\
#!/usr/bin/env bash
# Stage 1: GUI Grounding fine-tune
# Gate: ScreenSpot Acc@0.5 >= 75% before moving to Stage 2
set -euo pipefail
echo "Starting Stage 1 — GUI Grounding"
uv run llamafactory-cli train configs/training/sft_3b.yaml
echo "Stage 1 complete. Run: python training/evaluation/eval_screenspot.py"
"""

FILE_CONTENTS["training/scripts/train_stage2.sh"] = """\
#!/usr/bin/env bash
# Stage 2: Action Prediction fine-tune
# Gate: Action type accuracy >= 85%
set -euo pipefail
echo "Starting Stage 2 — Action Prediction"
uv run llamafactory-cli train configs/training/sft_3b_stage2.yaml
echo "Stage 2 complete."
"""

FILE_CONTENTS["training/scripts/train_stage3.sh"] = """\
#!/usr/bin/env bash
# Stage 3: Multi-Step Planning fine-tune
# Gate: ScreenSpot >= 79.6%, Mind2Web Step SR >= 44%
set -euo pipefail
echo "Starting Stage 3 — Multi-Step Planning"
uv run llamafactory-cli train configs/training/sft_3b_stage3.yaml
echo "Stage 3 complete. Run full eval: make eval"
"""

FILE_CONTENTS["training/scripts/export_model.sh"] = """\
#!/usr/bin/env bash
# Export fine-tuned model: merge LoRA → quantize → package
set -euo pipefail
CHECKPOINT="${1:-checkpoints/stage3_planning}"
OUTPUT_DIR="models/visionnav-3b"
echo "Exporting from: $CHECKPOINT"
echo "Output to     : $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
# TODO Phase 8: llamafactory-cli export + AWQ/GGUF quantization
echo "(stub) Export pipeline — implement in Phase 8"
"""

# ─── DOCKER FILES ────────────────────────────────────────────────────────────

FILE_CONTENTS["docker/Dockerfile.api"] = """\
# VisionNav API Service — no GPU required
FROM python:3.12-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project --no-dev

FROM deps AS app
COPY src/     ./src/
COPY configs/ ./configs/

ENV PYTHONPATH=/app/src
ENV VISIONNAV_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uv", "run", "uvicorn", "visionnav.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""

FILE_CONTENTS["docker/Dockerfile.inference"] = """\
# VisionNav Inference Service — requires NVIDIA GPU
# Models are mounted as volumes, not baked in (hot-swap without rebuild)
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "vllm>=0.8.4" "qwen-vl-utils>=0.0.11"

VOLUME /models

ENV MODEL_PATH=/models/visionnav-3b
ENV SERVED_MODEL_NAME=visionnav-3b
ENV MAX_MODEL_LEN=4096
ENV TENSOR_PARALLEL_SIZE=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD vllm serve ${MODEL_PATH} \
    --port 8001 \
    --served-model-name ${SERVED_MODEL_NAME} \
    --max-model-len ${MAX_MODEL_LEN} \
    --limit-mm-per-prompt image=3 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype bfloat16
"""

FILE_CONTENTS["docker/docker-compose.dev.yml"] = """\
# VisionNav local development stack
services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ../src:/app/src            # live reload
      - ../configs:/app/configs
      - screenshots:/tmp/visionnav/screenshots
    environment:
      - VISIONNAV_ENV=development
      - VISIONNAV_MODEL__BACKEND=vllm
      - VISIONNAV_MODEL__VLLM__BASE_URL=http://inference:8001
      - VISIONNAV_API__VALID_KEYS=["dev-key-123"]
      - LOG_LEVEL=DEBUG
    depends_on:
      inference:
        condition: service_healthy
    restart: unless-stopped

  inference:
    build:
      context: ..
      dockerfile: docker/Dockerfile.inference
    ports:
      - "8001:8001"
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # Uncomment for Phase 7+ scaling:
  # redis:
  #   image: redis:7-alpine
  #   ports: ["6379:6379"]
  #
  # db:
  #   image: postgres:16-alpine
  #   environment: { POSTGRES_DB: visionnav, POSTGRES_PASSWORD: localdev }
  #   ports: ["5432:5432"]

volumes:
  screenshots:
"""

FILE_CONTENTS["docker/docker-compose.prod.yml"] = """\
# VisionNav production stack
# All secrets via environment variables — nothing hardcoded here
services:
  api:
    image: visionnav-api:${IMAGE_TAG:-latest}
    ports:
      - "8000:8000"
    environment:
      - VISIONNAV_ENV=production
      - VISIONNAV_MODEL__BACKEND=vllm
      - VISIONNAV_MODEL__VLLM__BASE_URL=http://inference:8001
      - VISIONNAV_API__VALID_KEYS=${API_VALID_KEYS}
      - VISIONNAV_DB__URL=${DB_URL}
    depends_on:
      inference:
        condition: service_healthy
    restart: always

  inference:
    image: visionnav-inference:${IMAGE_TAG:-latest}
    volumes:
      - /data/models:/models
    environment:
      - MODEL_PATH=/models/visionnav-3b
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: always
"""

# ─── DOCS FILES ──────────────────────────────────────────────────────────────

FILE_CONTENTS["docs/api.md"] = """\
# VisionNav API Reference

## Base URL
- Development : `http://localhost:8000`
- Production  : `https://api.visionnav.ai`

## Authentication

## Endpoints

### POST /v1/tasks/
Submit a task. Returns immediately — poll for status.
```json
{ "instruction": "Open Chrome and search for AI news", "platform": "desktop", "max_steps": 50 }
```
Response `202`:
```json
{ "task_id": "a1b2c3...", "status": "accepted", "message": "Poll GET /v1/tasks/{task_id}" }
```

### GET /v1/tasks/{task_id}
Get task status and step history.

### GET /v1/health
Liveness — returns `{"status": "ok"}`.

### GET /v1/ready
Readiness — returns `{"status": "ok"}` when model is loaded.

## Interactive Docs
Visit `http://localhost:8000/docs` (development mode only).
"""

FILE_CONTENTS["docs/training.md"] = """\
# VisionNav Training Guide

## Prerequisites
```bash
uv sync --extra train
export WANDB_API_KEY=your_key
export WANDB_PROJECT=visionnav
```

## 3-Stage Training Pipeline

### Stage 1 — GUI Grounding
**Target:** ScreenSpot Acc@0.5 >= 75%
```bash
make train-stage1
python training/evaluation/eval_screenspot.py --model checkpoints/stage1_best --threshold 75.0
```

### Stage 2 — Action Prediction
**Target:** Action type accuracy >= 85%
```bash
make train-stage2
```

### Stage 3 — Multi-Step Planning
**Target:** ScreenSpot >= 79.6%, Mind2Web Step SR >= 44%
```bash
make train-stage3
make eval
```

## Dataset Preparation
```bash
make data-all
```

## Benchmark Performance Targets
| Stage    | ScreenSpot | Mind2Web SR | Reference          |
|----------|-----------|-------------|-------------------|
| Stage 1  | >= 75.0%  | —           | Grounding baseline |
| Stage 2  | >= 79.6%  | >= 44%      | TongUI-3B level    |
| Stage 3  | >= 83.6%  | >= 53%      | TongUI-3B-1M level |
| 7B model | >= 86.0%  | >= 55%      | TongUI-7B-1M level |
"""

FILE_CONTENTS["docs/deployment.md"] = """\
# VisionNav Deployment Guide

## Local Development (no Docker)
```bash
# Terminal 1 — vLLM inference server (needs GPU)
make serve-vllm

# Terminal 2 — FastAPI server
make serve-api
```

## Local Development (Docker)
```bash
cp .env.example .env   # fill in your values
make docker-dev
```

## Environment Variables
All config is environment-driven. Copy `.env.example` → `.env`.

Key variables:
| Variable | Dev default | Production |
|---|---|---|
| `VISIONNAV_MODEL__BACKEND` | `local` | `vllm` |
| `VISIONNAV_MODEL__VLLM__BASE_URL` | `http://localhost:8001` | `http://inference:8001` |
| `VISIONNAV_API__VALID_KEYS` | `["dev-key-123"]` | Set via secrets manager |
| `VISIONNAV_DB__URL` | SQLite path | `postgresql+asyncpg://...` |

## Switch SQLite → PostgreSQL
Change **one environment variable** — zero code changes:
```bash
VISIONNAV_DB__URL=postgresql+asyncpg://user:pass@host:5432/visionnav
```

## Switch local model → vLLM
```bash
VISIONNAV_MODEL__BACKEND=vllm
VISIONNAV_MODEL__VLLM__BASE_URL=http://your-gpu-server:8001
```

## Production (AWS)
- API → ECS Fargate (no GPU, t3.medium+)
- Inference → ECS with GPU instance (g5.xlarge, 24GB VRAM)
- Database → RDS PostgreSQL (change DB_URL only)
- Storage → S3 (change VISIONNAV_AGENT__SCREENSHOT_DIR)
"""

# ─── NOTEBOOKS ───────────────────────────────────────────────────────────────

FILE_CONTENTS["notebooks/01_dataset_exploration.ipynb"] = """\
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# GUI-Net-1M Dataset Exploration\\n",
              "Analyse structure, quality, action distribution, and resolution spread."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["from datasets import load_dataset\\n",
              "ds = load_dataset('Bofeee5675/GUI-Net-1M', split='train', streaming=True)\\n",
              "sample = next(iter(ds))\\n",
              "print(sample.keys())"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""
FILE_CONTENTS["README.md"] = """\
# VisionNav

> AI-powered GUI navigation agent — sees your screen, understands UI elements,
> and controls mouse/keyboard to complete tasks from natural language instructions.

## Quick Start
```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup dev environment
make dev

# 3. Copy and configure environment
cp .env.example .env

# 4. Run tests
make test-unit

# 5. Start API server
make serve-api
```

## Documentation
- [API Reference](docs/api.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)

## Architecture
Built on Qwen2.5-VL-3B fine-tuned on GUI-Net-1M.
See [VisionNav MVP Architecture](docs/deployment.md) for full details.

## License
MIT
"""

FILE_CONTENTS[".python-version"] = "3.12\n"

FILE_CONTENTS["configs/base.yaml"] = """\
agent:
  max_steps: 50
  screenshot_dir: "/tmp/visionnav/screenshots"
  change_threshold: 0.01

model:
  backend: "local"
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  dtype: "bfloat16"
  device_map: "auto"
  vllm:
    base_url: "http://localhost:8001"
    model_name: "visionnav-3b"
    timeout_seconds: 30

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  valid_keys: []

db:
  url: "sqlite+aiosqlite:///./visionnav.db"

platforms:
  desktop:
    capture:
      monitor_index: 1
    mouse:
      move_duration: 0.15
      click_delay: 0.1
      use_bezier_path: true
    keyboard:
      typing_wpm: 60
      key_delay: 0.05
  android:
    adb:
      device_serial: null
      screencap_method: "adb"
    touch:
      tap_duration: 50
      swipe_duration: 300
"""

FILE_CONTENTS["configs/development.yaml"] = """\
# Development overrides — inherits from base.yaml
agent:
  max_steps: 20

model:
  backend: "local"

api:
  valid_keys: ["dev-key-123"]

db:
  url: "sqlite+aiosqlite:///./visionnav_dev.db"
"""

FILE_CONTENTS["configs/production.yaml"] = """\
# Production overrides — secrets come from environment variables, NOT here
agent:
  max_steps: 30

model:
  backend: "vllm"

api:
  cors_origins:
    - "https://visionnav.ai"
    - "https://app.visionnav.ai"
"""

FILE_CONTENTS["configs/training/sft_3b.yaml"] = """\
### Stage 1 — GUI Grounding
### Gate: ScreenSpot Acc@0.5 >= 75% before Stage 2
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj
dataset: gui_video_full,wikihow_v3,showui-desktop-augmented,showui-web
template: qwen2_vl
cutoff_len: 4096
max_samples: 999999
overwrite_cache: true
preprocessing_num_workers: 8
output_dir: checkpoints/stage1_grounding
logging_steps: 10
save_steps: 500
eval_steps: 500
plot_loss: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
report_to: wandb
run_name: visionnav-3b-stage1
"""

FILE_CONTENTS["configs/training/sft_3b_stage2.yaml"] = """\
### Stage 2 — Action Prediction
### Gate: Action type accuracy >= 85%
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj
resume_from_checkpoint: checkpoints/stage1_grounding
dataset: aitw_with_thoughts,mind2web_with_thoughts,guiact_smartphone,miniwob_with_thoughts
template: qwen2_vl
cutoff_len: 4096
preprocessing_num_workers: 8
output_dir: checkpoints/stage2_action
logging_steps: 10
save_steps: 500
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
report_to: wandb
run_name: visionnav-3b-stage2
"""

FILE_CONTENTS["configs/training/sft_3b_stage3.yaml"] = """\
### Stage 3 — Multi-Step Planning
### Gate: ScreenSpot >= 79.6%, Mind2Web SR >= 44%
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
resume_from_checkpoint: checkpoints/stage2_action
dataset: visionnav_multistep,visionnav_recovery,baidu_jingyan_train,wikihow_v3
template: qwen2_vl
cutoff_len: 8192
preprocessing_num_workers: 8
output_dir: checkpoints/stage3_planning
logging_steps: 10
save_steps: 500
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
report_to: wandb
run_name: visionnav-3b-stage3
"""

FILE_CONTENTS["data/.gitkeep"] = (
    "# Dataset contents managed by DVC. Run: make data-download\n"
)

FILE_CONTENTS["data/dataset_info.json"] = """\
{
  "_comment": "LLaMA-Factory dataset registry",
  "gui_video_full":         {"file_name": "tongui/gui_video_full.json",        "formatting": "sharegpt"},
  "wikihow_v3":             {"file_name": "tongui/wikihow_v3.json",             "formatting": "sharegpt"},
  "baidu_jingyan_train":    {"file_name": "tongui/baidu_jingyan_train.json",    "formatting": "sharegpt"},
  "aitw_with_thoughts":     {"file_name": "external/aitw_with_thoughts.json",   "formatting": "sharegpt"},
  "mind2web_with_thoughts": {"file_name": "external/mind2web_with_thoughts.json","formatting": "sharegpt"},
  "guiact_smartphone":      {"file_name": "external/guiact_smartphone.json",    "formatting": "sharegpt"},
  "miniwob_with_thoughts":  {"file_name": "external/miniwob_with_thoughts.json","formatting": "sharegpt"},
  "showui-desktop-augmented":{"file_name": "external/showui_desktop.json",      "formatting": "sharegpt"},
  "showui-web":             {"file_name": "external/showui_web.json",            "formatting": "sharegpt"},
  "visionnav_multistep":    {"file_name": "visionnav/multistep_tasks.json",     "formatting": "sharegpt"},
  "visionnav_recovery":     {"file_name": "visionnav/recovery_scenarios.json",  "formatting": "sharegpt"},
  "visionnav_windows":      {"file_name": "visionnav/windows_tasks.json",       "formatting": "sharegpt"},
  "visionnav_android":      {"file_name": "visionnav/android_tasks.json",       "formatting": "sharegpt"}
}
"""

FILE_CONTENTS["src/visionnav/memory/models.py"] = """\
\"\"\"
SQLModel ORM models for memory persistence.
Currently unused in MVP (sqlite.py uses raw aiosqlite).
Ready for Phase 7+ when we switch to SQLModel + PostgreSQL.
\"\"\"
from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class Task(SQLModel, table=True):
    task_id:     str           = Field(primary_key=True)
    instruction: str
    status:      str           = "running"
    result:      Optional[str] = None
    created_at:  datetime      = Field(default_factory=datetime.utcnow)


class Step(SQLModel, table=True):
    id:          Optional[int] = Field(default=None, primary_key=True)
    task_id:     str           = Field(foreign_key="task.task_id")
    step_index:  int
    action_type: Optional[str] = None
    success:     bool          = False
    error:       Optional[str] = None
    created_at:  datetime      = Field(default_factory=datetime.utcnow)
"""
FILE_CONTENTS["tests/unit/test_action_schema.py"] = """\
\"\"\"Unit tests — Action schema validation.\"\"\"
import pytest
from pydantic import ValidationError
from visionnav.actions.schema import Action, ActionType


def test_valid_click():
    a = Action(type=ActionType.CLICK, coordinates=(0.5, 0.5))
    assert a.type == ActionType.CLICK

def test_confidence_bounds():
    with pytest.raises(ValidationError):
        Action(type=ActionType.CLICK, confidence=1.5)
    with pytest.raises(ValidationError):
        Action(type=ActionType.CLICK, confidence=-0.1)

def test_default_confidence():
    a = Action(type=ActionType.DONE)
    assert a.confidence == 1.0

def test_all_action_types_valid():
    for at in ActionType:
        a = Action(type=at)
        assert a.type == at
"""

FILE_CONTENTS["tests/unit/test_ocr_postprocess.py"] = """\
\"\"\"Unit tests — OCR output validation and filtering.\"\"\"
from visionnav.perception.ocr import TextRegion


def test_text_region_fields():
    r = TextRegion(text="Submit", bbox=(0.1, 0.1, 0.4, 0.15), confidence=0.95)
    assert r.text == "Submit"
    assert r.confidence == 0.95

def test_bbox_normalised():
    r = TextRegion(text="OK", bbox=(0.0, 0.0, 1.0, 1.0), confidence=0.8)
    x1, y1, x2, y2 = r.bbox
    assert 0.0 <= x1 <= 1.0
    assert 0.0 <= y1 <= 1.0
    assert x2 >= x1
    assert y2 >= y1

def test_low_confidence_check():
    r = TextRegion(text="blurry", bbox=(0.0, 0.0, 0.1, 0.05), confidence=0.3)
    assert r.confidence < 0.5   # Would be filtered by OCREngine min_confidence
"""

FILE_CONTENTS["tests/integration/test_platform_capture.py"] = """\
\"\"\"Integration tests — screen capture (runs on real OS).\"\"\"
import pytest
import numpy as np


@pytest.mark.integration
def test_capture_returns_rgb_array():
    \"\"\"Must run on a real machine with a display.\"\"\"
    from visionnav.perception.capture import ScreenCapture
    cap  = ScreenCapture()
    arr, meta = cap.capture()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
    assert arr.shape[2] == 3          # RGB channels
    assert meta["width"]  > 100
    assert meta["height"] > 100

@pytest.mark.integration
def test_capture_size_matches_metadata():
    from visionnav.perception.capture import ScreenCapture
    cap  = ScreenCapture()
    arr, meta = cap.capture()
    assert arr.shape[1] == meta["width"]
    assert arr.shape[0] == meta["height"]

@pytest.mark.integration
def test_screen_size_consistent():
    from visionnav.perception.capture import ScreenCapture
    cap = ScreenCapture()
    w, h = cap.get_screen_size()
    assert w > 0 and h > 0
"""

FILE_CONTENTS["tests/e2e/test_open_chrome.py"] = """\
\"\"\"E2E test — open Chrome and verify (sandboxed environment required).\"\"\"
import pytest


@pytest.mark.e2e
async def test_open_chrome_completes():
    \"\"\"
    Full agent run in isolated environment.
    Requires: sandbox VM or container with Chrome installed.
    Implement in Phase 6 after agent loop is stable.
    \"\"\"
    pytest.skip("E2E tests enabled in Phase 6")
"""

FILE_CONTENTS["tests/e2e/test_fill_form.py"] = """\
\"\"\"E2E test — fill a web form (sandboxed environment required).\"\"\"
import pytest


@pytest.mark.e2e
async def test_fill_contact_form():
    \"\"\"
    Full agent run filling a test form.
    Implement in Phase 6 after agent loop is stable.
    \"\"\"
    pytest.skip("E2E tests enabled in Phase 6")
"""
FILE_CONTENTS["scripts/setup_dev.sh"] = """\
#!/usr/bin/env bash
# One-command dev environment setup
set -euo pipefail

echo "Setting up VisionNav dev environment..."

# Check uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Sync all dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Copy .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — add your tokens!"
fi

echo "Done! Run: make test-unit"
"""

FILE_CONTENTS["scripts/lint.sh"] = """\
#!/usr/bin/env bash
# Run all linting and type checking
set -euo pipefail
echo "Running ruff..."
uv run ruff check src/ tests/ data_pipeline/
uv run ruff format --check src/ tests/
echo "Running mypy..."
uv run mypy src/visionnav --ignore-missing-imports
echo "All checks passed!"
"""

FILE_CONTENTS["scripts/generate_openapi.sh"] = """\
#!/usr/bin/env bash
# Export OpenAPI spec to docs/openapi.yaml
set -euo pipefail
uv run python -c "
import yaml, json
from visionnav.api.app import create_app
app    = create_app()
schema = app.openapi()
with open('docs/openapi.yaml', 'w') as f:
    yaml.dump(schema, f, default_flow_style=False)
print('OpenAPI spec written to docs/openapi.yaml')
"
"""
FILE_CONTENTS["docs/openapi.yaml"] = (
    "# Auto-generated. Run: bash scripts/generate_openapi.sh\n"
)

FILE_CONTENTS["notebooks/02_ocr_quality.ipynb"] = """\
{
 "cells": [
  {"cell_type":"markdown","metadata":{},
   "source":["# OCR Quality Analysis\\n","Measure PaddleOCR vs Tesseract accuracy on GUI screenshots."]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],
   "source":["from visionnav.perception.ocr import OCREngine\\n",
             "from PIL import Image\\n","import numpy as np\\n",
             "\\n","ocr = OCREngine()\\n",
             "# Load a test screenshot and run OCR\\n",
             "# img = np.array(Image.open('tests/fixtures/sample_screen.png'))\\n",
             "# regions = ocr.run(img)\\n","# for r in regions: print(r)"]}
 ],
 "metadata":{
  "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
  "language_info":{"name":"python","version":"3.12.0"}
 },
 "nbformat":4,"nbformat_minor":4
}
"""

FILE_CONTENTS["notebooks/03_model_baseline.ipynb"] = """\
{
 "cells": [
  {"cell_type":"markdown","metadata":{},
   "source":["# Model Baseline Evaluation\\n",
             "Run Qwen2.5-VL-3B (no fine-tuning) on sample GUI screenshots.\\n",
             "This establishes the baseline before Stage 1 training."]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],
   "source":["# from visionnav.models.local import LocalModelBackend\\n",
             "# from visionnav.settings import ModelSettings\\n",
             "# backend = LocalModelBackend(ModelSettings())\\n",
             "# Test with a sample screenshot..."]}
 ],
 "metadata":{
  "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
  "language_info":{"name":"python","version":"3.12.0"}
 },
 "nbformat":4,"nbformat_minor":4
}
"""


# ─── RUNNER FUNCTIONS (required — script won't work without these) ────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def create_structure(base: Path) -> tuple[int, int, int]:
    created = skipped = dirs = 0

    for d in EMPTY_DIRS:
        p = base / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            dirs += 1

    for rel, content in FILE_CONTENTS.items():
        fp = base / rel
        if not fp.parent.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
            dirs += 1
        if fp.exists():
            print(f"  {YELLOW}SKIP  {RESET}{rel}")
            skipped += 1
            continue
        fp.write_text(content, encoding="utf-8")
        print(f"  {GREEN}CREATE{RESET} {rel}")
        created += 1

    return created, skipped, dirs


def print_banner() -> None:
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║       VisionNav Project Structure Generator              ║
║       github.com/AdnanKhaan11/visionnav                  ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")


def print_summary(base: Path, created: int, skipped: int, dirs: int) -> None:
    print(f"""
{BOLD}{'='*58}
Summary
{'='*58}{RESET}
  {GREEN}✓ Files created  : {created}{RESET}
  {YELLOW}⚠ Files skipped  : {skipped} (already exist — safe){RESET}
  {CYAN}  Dirs created   : {dirs}{RESET}
  {RESET}  Location       : {base.resolve()}

{BOLD}Next Steps:{RESET}

  1. {CYAN}Install uv (if not already){RESET}
     curl -LsSf https://astral.sh/uv/install.sh | sh

  2. {CYAN}Set up dev environment{RESET}
     make dev

  3. {CYAN}Configure environment{RESET}
     cp .env.example .env
     # edit .env — add HF_TOKEN, WANDB_API_KEY etc.

  4. {CYAN}Run unit tests to verify setup{RESET}
     make test-unit

  5. {CYAN}Start API server (local model){RESET}
     make serve-api

  6. {CYAN}Download dataset (Phase 1){RESET}
     make data-download

{BOLD}Docs:{RESET}
  docs/api.md         API reference
  docs/training.md    Training pipeline
  docs/deployment.md  Deployment guide

{GREEN}VisionNav is ready. Start building!{RESET}
""")


def main() -> None:
    import sys

    print_banner()

    base = Path.cwd()

    if not (base / ".git").exists():
        print(f"{YELLOW}Warning: no .git found in {base}{RESET}")
        print(f"{YELLOW}Make sure you are INSIDE your cloned visionnav repo.{RESET}")
        ans = input("\n  Continue anyway? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"\n{BOLD}Creating structure in:{RESET} {CYAN}{base.resolve()}{RESET}\n")
    created, skipped, dirs = create_structure(base)
    print_summary(base, created, skipped, dirs)


if __name__ == "__main__":
    main()
