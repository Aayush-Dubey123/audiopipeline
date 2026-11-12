"""
Shared test configuration.

The suite runs against a throwaway database and storage folder so tests never write job
folders into backend/storage/tmp or rows into the live voice_studio.db. This must happen
before any `core` import, because settings and the DB engine are created at import time.
"""

import os
import shutil
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="voice_studio_tests_"))
os.environ["TMP_DIR"] = str(_TEST_ROOT / "tmp")
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}"

import pytest  # noqa: E402

from core.apis import middleware  # noqa: E402


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


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Remove the throwaway database and storage folder."""
    try:
        from core.database.database import engine

        engine.dispose()
    except Exception:
        pass
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
