"""E2E test — open Chrome and verify (sandboxed environment required)."""
import pytest


@pytest.mark.e2e
async def test_open_chrome_completes():
    """
    Full agent run in isolated environment.
    Requires: sandbox VM or container with Chrome installed.
    Implement in Phase 6 after agent loop is stable.
    """
    pytest.skip("E2E tests enabled in Phase 6")
