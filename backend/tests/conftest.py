"""
Shared test configuration.
"""

import pytest

from core.apis import middleware


@pytest.fixture(autouse=True)
def _relaxed_rate_limit(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """
    The suite fires many POSTs from one TestClient inside a minute. Raise the per-IP
    limit for functional tests; tests marked `rate_limit` keep the real limit.
    """
    if request.node.get_closest_marker("rate_limit") is None:
        monkeypatch.setattr(middleware.settings, "RATE_LIMIT_PER_MINUTE", 10_000)
    yield


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "rate_limit: keep the production rate limit for this test")
