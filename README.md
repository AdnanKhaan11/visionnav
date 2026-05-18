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
