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
