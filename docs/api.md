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


==========================================
PS D:\visionnav> python setup_visionnav.py

╔══════════════════════════════════════════════════════════╗
║       VisionNav Project Structure Generator              ║
║       github.com/AdnanKhaan11/visionnav                  ║
╚══════════════════════════════════════════════════════════╝


Creating structure in: D:\visionnav

  CREATE src/visionnav/__init__.py
  CREATE src/visionnav/settings.py
  CREATE src/visionnav/agent/__init__.py
  CREATE src/visionnav/agent/state.py
  CREATE src/visionnav/agent/planner.py
  CREATE src/visionnav/agent/reporter.py
  CREATE src/visionnav/agent/agent.py
  CREATE src/visionnav/perception/__init__.py
  CREATE src/visionnav/perception/capture.py
  CREATE src/visionnav/perception/ocr.py
  CREATE src/visionnav/perception/ui_tree.py
  CREATE src/visionnav/perception/fusion.py
  CREATE src/visionnav/models/__init__.py
  CREATE src/visionnav/models/base.py
  CREATE src/visionnav/models/prompt.py
  CREATE src/visionnav/models/local.py
  CREATE src/visionnav/models/vllm_backend.py
  CREATE src/visionnav/actions/__init__.py
  CREATE src/visionnav/actions/schema.py
  CREATE src/visionnav/actions/parser.py
  CREATE src/visionnav/actions/executor.py
  CREATE src/visionnav/actions/verifier.py
  CREATE src/visionnav/platforms/__init__.py
  CREATE src/visionnav/platforms/base.py
  CREATE src/visionnav/platforms/desktop.py
  CREATE src/visionnav/platforms/android.py
  CREATE src/visionnav/memory/__init__.py
  CREATE src/visionnav/memory/base.py
  CREATE src/visionnav/memory/sqlite.py
  CREATE src/visionnav/safety/__init__.py
  CREATE src/visionnav/safety/classifier.py
  CREATE src/visionnav/safety/gates.py
  CREATE src/visionnav/api/__init__.py
  CREATE src/visionnav/api/app.py
  CREATE src/visionnav/api/dependencies.py
  CREATE src/visionnav/api/middleware.py
  CREATE src/visionnav/api/v1/__init__.py
  CREATE src/visionnav/api/v1/router.py
  CREATE src/visionnav/api/v1/health.py
  CREATE src/visionnav/api/v1/tasks.py
  CREATE src/visionnav/api/v1/sessions.py
  CREATE src/visionnav/utils/__init__.py
  CREATE src/visionnav/utils/image.py
  CREATE src/visionnav/utils/coords.py
  CREATE src/visionnav/utils/retry.py
  CREATE src/visionnav/utils/logging.py
  CREATE tests/conftest.py
  CREATE tests/unit/test_action_parser.py
  CREATE tests/unit/test_coord_utils.py
  CREATE tests/unit/test_action_verifier.py
  CREATE tests/unit/test_safety_classifier.py
  CREATE tests/unit/test_prompt_builder.py
  CREATE tests/integration/test_agent_loop.py
  CREATE tests/integration/test_api_tasks.py
  CREATE data_pipeline/__init__.py
  CREATE data_pipeline/pipeline.py
  CREATE data_pipeline/downloaders.py
  CREATE data_pipeline/cleaners.py
  CREATE data_pipeline/ocr_enricher.py
  CREATE data_pipeline/formatters.py
  CREATE data_pipeline/validators.py
  CREATE training/evaluation/eval_screenspot.py
  CREATE training/evaluation/eval_mind2web.py
  CREATE training/evaluation/eval_visionnav.py
  CREATE training/scripts/prepare_data.sh
  CREATE training/scripts/train_stage1.sh
  CREATE training/scripts/train_stage2.sh
  CREATE training/scripts/train_stage3.sh
  CREATE training/scripts/export_model.sh
  CREATE docker/Dockerfile.api
  CREATE docker/Dockerfile.inference
  CREATE docker/docker-compose.dev.yml
  CREATE docker/docker-compose.prod.yml
  CREATE docs/api.md
  CREATE docs/training.md
  CREATE docs/deployment.md
  CREATE notebooks/01_dataset_exploration.ipynb

==========================================================
Summary
==========================================================
  ✓ Files created  : 77
  ⚠ Files skipped  : 0 (already exist — safe)
    Dirs created   : 33
    Location       : D:\visionnav

Next Steps:

  1. Install uv (if not already)
     curl -LsSf https://astral.sh/uv/install.sh | sh

  2. Set up dev environment
     make dev

  3. Configure environment
     cp .env.example .env
     # edit .env — add HF_TOKEN, WANDB_API_KEY etc.

  4. Run unit tests to verify setup
     make test-unit

  5. Start API server (local model)
     make serve-api

  6. Download dataset (Phase 1)
     make data-download

Docs:
  docs/api.md         API reference
  docs/training.md    Training pipeline
  docs/deployment.md  Deployment guide

VisionNav is ready. Start building!

PS D:\visionnav> 