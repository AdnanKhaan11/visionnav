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
