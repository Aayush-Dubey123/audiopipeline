"""
Cleanup service for managing temporary inference storage and expired job retention.

Keeps storage/tmp bounded:
  * completed jobs keep exactly two audio files (output.wav + output.mp3) until TTL,
  * failed jobs keep resumable chunks only for FAILED_JOB_TTL_MINUTES,
  * folders with no database record (crashes, interrupted uploads) are removed,
  * the speaker-conditioning cache is capped to CONDS_CACHE_MAX_FILES entries.
"""

import re
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Set

from sqlalchemy import select
from sqlalchemy.orm import Session

from core import logger
from core.config import get_settings
from core.cruds.job_crud import JobCRUD
from core.models.job_model import JobModel

logging = logger(__name__)
settings = get_settings()

# Files a finished job is allowed to keep.
FINAL_OUTPUT_FILES = {"output.wav", "output.mp3"}
FINAL_PREVIEW_FILES = {"preview.wav"}

# Directories inside tmp that are not job folders.
_RESERVED_DIRS = {"cache", "reference_checks"}
_JOB_DIR_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

# Grace period so the janitor never races a request that is still writing an upload.
_ORPHAN_GRACE_SECONDS = 15 * 60


class CleanupService:
    """
    Automated lifecycle and artifact purge service for inference files.
    """

    @staticmethod
    def _safe_job_dir(job_id: str) -> Path | None:
        base = settings.tmp_path.resolve()
        job_dir = (base / job_id).resolve()
        try:
            job_dir.relative_to(base)
        except ValueError:
            logging.error(f"Security error: path traversal attempt in cleanup for job '{job_id}'")
            return None
        if job_dir == base:
            return None
        return job_dir

    @classmethod
    def cleanup_job_directory(cls, job_id: str) -> bool:
        """
        Delete the physical temporary working directory for a given job.

        Args:
            job_id (str): Unique identifier of target job.

        Returns:
            bool: True if folder existed and was removed, False otherwise.
        """
        logging.info(f"Executing CleanupService.cleanup_job_directory for job '{job_id}'")
        job_dir = cls._safe_job_dir(job_id)
        if job_dir is None:
            return False

        if job_dir.exists() and job_dir.is_dir():
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
                logging.info(f"Purged artifacts for job '{job_id}' at {job_dir}")
                return True
            except Exception as error:
                logging.warning(f"Error purging artifacts for job '{job_id}': {error}")
                return False
        return False

    @classmethod
    def keep_only_final_outputs(cls, job_dir: Path, keep: Set[str] = FINAL_OUTPUT_FILES) -> int:
        """
        Remove every file/folder in a finished job directory except the deliverables.

        Args:
            job_dir (Path): Job working directory.
            keep (Set[str]): File names to keep.

        Returns:
            int: Number of entries removed.
        """
        removed = 0
        if not job_dir.is_dir():
            return 0
        for entry in job_dir.iterdir():
            if entry.name in keep and entry.is_file():
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                removed += 1
            except OSError as err:
                logging.warning(f"Could not remove intermediate file {entry}: {err}")
        return removed

    @classmethod
    def clean_expired_jobs(cls, db: Session) -> int:
        """
        Scan database for expired jobs, purge corresponding audio files, and transition state.

        Args:
            db (Session): Active database session.

        Returns:
            int: Number of expired jobs cleaned.
        """
        logging.info("Executing CleanupService.clean_expired_jobs")
        now = datetime.now(timezone.utc)
        try:
            expired_jobs = JobCRUD.get_expired_jobs(db=db, reference_time=now)
            count = 0
            for job in expired_jobs:
                cls.cleanup_job_directory(job.id)
                JobCRUD.update_job_status(
                    db=db,
                    job_id=job.id,
                    status="expired",
                    progress_message="Job artifacts have expired and were purged.",
                )
                count += 1
            if count > 0:
                logging.info(f"CleanupService cleaned {count} expired job(s)")
            return count
        except Exception as error:
            logging.error(f"Error in CleanupService.clean_expired_jobs: {error}")
            return 0

    @classmethod
    def clean_stale_failed_jobs(cls, db: Session) -> int:
        """
        Purge resumable chunks of failed jobs older than FAILED_JOB_TTL_MINUTES.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=settings.FAILED_JOB_TTL_MINUTES)
        count = 0
        try:
            rows = db.execute(
                select(JobModel.id, JobModel.updated_at).where(JobModel.status == "failed")
            ).all()
            for job_id, updated_at in rows:
                if updated_at is None:
                    continue
                ts = updated_at if updated_at.tzinfo else updated_at.replace(tzinfo=timezone.utc)
                if ts <= cutoff and cls.cleanup_job_directory(job_id):
                    count += 1
        except Exception as error:
            logging.error(f"Error in CleanupService.clean_stale_failed_jobs: {error}")
        return count

    @classmethod
    def clean_orphan_dirs(cls, db: Session) -> int:
        """
        Remove tmp folders with no live job record (crash leftovers, aborted uploads).
        """
        base = settings.tmp_path
        if not base.is_dir():
            return 0
        now = time.time()
        count = 0
        try:
            live_ids = {
                row[0]
                for row in db.execute(
                    select(JobModel.id).where(JobModel.status != "expired")
                ).all()
            }
        except Exception as error:
            logging.error(f"Orphan scan skipped, could not read jobs: {error}")
            return 0

        for entry in base.iterdir():
            if entry.name in _RESERVED_DIRS:
                continue
            try:
                age = now - entry.stat().st_mtime
            except OSError:
                continue
            if age < _ORPHAN_GRACE_SECONDS:
                continue
            if entry.is_dir() and _JOB_DIR_RE.fullmatch(entry.name) and entry.name in live_ids:
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                count += 1
            except OSError as err:
                logging.warning(f"Could not remove orphan {entry}: {err}")

        # Stray uploads left by an interrupted reference check
        checks = base / "reference_checks"
        if checks.is_dir():
            for f in checks.iterdir():
                try:
                    if now - f.stat().st_mtime > _ORPHAN_GRACE_SECONDS:
                        f.unlink(missing_ok=True)
                        count += 1
                except OSError:
                    pass
        return count

    @staticmethod
    def trim_conditionals_cache() -> int:
        """
        Keep only the most recently used speaker-conditioning cache files.
        """
        cache_dir = settings.tmp_path / "cache" / "conditionals"
        if not cache_dir.is_dir():
            return 0
        files = sorted(cache_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for stale in files[max(settings.CONDS_CACHE_MAX_FILES, 0):]:
            stale.unlink(missing_ok=True)
            removed += 1
        return removed

    @classmethod
    def run_janitor(cls, db: Session) -> dict:
        """
        Run every cleanup pass. Safe to call at startup and periodically.
        """
        result = {
            "expired": cls.clean_expired_jobs(db),
            "failed": cls.clean_stale_failed_jobs(db),
            "orphans": cls.clean_orphan_dirs(db),
            "cache": cls.trim_conditionals_cache(),
        }
        if any(result.values()):
            logging.info(f"Janitor pass removed: {result}")
        return result
