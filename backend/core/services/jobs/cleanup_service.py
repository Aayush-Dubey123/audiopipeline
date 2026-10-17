"""
Cleanup service for managing temporary inference storage and expired job retention.

Enforces storage quotas by purging expired audio directories and setting expired states.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy.orm import Session

from core import logger
from core.config import get_settings
from core.cruds.job_crud import JobCRUD

logging = logger(__name__)
settings = get_settings()


class CleanupService:
    """
    Automated lifecycle and artifact purge service for inference files.
    """

    @staticmethod
    def cleanup_job_directory(job_id: str) -> bool:
        """
        Delete the physical temporary working directory for a given job.

        Args:
            job_id (str): Unique identifier of target job.

        Returns:
            bool: True if folder existed and was removed, False otherwise.
        """
        logging.info(f"Executing CleanupService.cleanup_job_directory for job '{job_id}'")
        job_dir = (settings.tmp_path / job_id).resolve()
        try:
            job_dir.relative_to(settings.tmp_path.resolve())
        except ValueError:
            logging.error(f"Security error: path traversal attempt in cleanup for job '{job_id}'")
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
