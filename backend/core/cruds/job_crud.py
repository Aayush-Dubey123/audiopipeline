"""
Database CRUD repository for inference job persistence entities.

Provides transactional job lifecycle updates, retrieval, stale job recovery, and cleanup queries.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, update, func
from core import logger
from core.models.job_model import JobModel
from core.config import get_settings

logging = logger(__name__)
_settings = get_settings()


class JobCRUD:
    """
    Persistence operations for asynchronous job entities.
    """

    @staticmethod
    def create_job(db: Session, job_data: dict) -> JobModel:
        """
        Persist a newly initialized inference job record.

        Args:
            db (Session): Active database session.
            job_data (dict): Validated job attribute dictionary.

        Returns:
            JobModel: Newly created JobModel instance.

        Raises:
            Exception: If database commit fails.
        """
        logging.info(f"Executing JobCRUD.create_job with id '{job_data.get('id')}'")
        try:
            job = JobModel(**job_data)
            db.add(job)
            db.commit()
            db.refresh(job)
            return job
        except Exception as error:
            db.rollback()
            logging.error(f"Error in JobCRUD.create_job: {error}")
            raise error

    @staticmethod
    def get_job_by_id(db: Session, job_id: str) -> Optional[JobModel]:
        """
        Retrieve an inference job by its unguessable identifier.

        Args:
            db (Session): Active database session.
            job_id (str): Unique job identifier string.

        Returns:
            Optional[JobModel]: Job entity or None if record does not exist.
        """
        logging.info(f"Executing JobCRUD.get_job_by_id for job '{job_id}'")
        try:
            db.expire_all()
            query = select(JobModel).where(JobModel.id == job_id)
            return db.execute(query).scalar_one_or_none()
        except Exception as error:
            logging.error(f"Error in JobCRUD.get_job_by_id for job '{job_id}': {error}")
            raise error

    @staticmethod
    def update_job_status(
        db: Session,
        job_id: str,
        status: str,
        progress_message: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        output_metadata: Optional[dict] = None,
    ) -> Optional[JobModel]:
        """
        Update execution status and state transitions for a job.

        When status transitions to 'completed' or 'failed', stamps completed_at and
        sets expires_at to completed_at + JOB_TTL_MINUTES so retention counts from
        completion rather than creation.

        Args:
            db (Session): Active database session.
            job_id (str): Unique identifier of the target job.
            status (str): New status value (e.g. processing, completed, failed, expired).
            progress_message (Optional[str]): Informative progress message.
            error_code (Optional[str]): Error code if status is failed.
            error_message (Optional[str]): Error message if status is failed.
            output_metadata (Optional[dict]): Output metrics if completed.

        Returns:
            Optional[JobModel]: Updated job record or None if not found.
        """
        logging.info(f"Executing JobCRUD.update_job_status for job '{job_id}' -> '{status}'")
        try:
            job = db.execute(select(JobModel).where(JobModel.id == job_id)).scalar_one_or_none()
            if not job:
                logging.warning(f"Cannot update status for nonexistent job '{job_id}'")
                return None

            now = datetime.now(timezone.utc)
            job.status = status
            job.updated_at = now
            if progress_message is not None:
                job.progress_message = progress_message
            if error_code is not None:
                job.error_code = error_code
            if error_message is not None:
                job.error_message = error_message
            if output_metadata is not None:
                job.output_metadata = output_metadata

            # Reset error and completion state if requeued / resumed
            if status == "queued":
                job.error_code = None
                job.error_message = None
                job.completed_at = None

            # Stamp completion time and recalculate TTL from completion
            if status in ("completed", "failed") and job.completed_at is None:
                job.completed_at = now
                job.expires_at = now + timedelta(minutes=_settings.JOB_TTL_MINUTES)

            db.commit()
            db.refresh(job)
            return job
        except Exception as error:
            db.rollback()
            logging.error(f"Error in JobCRUD.update_job_status for job '{job_id}': {error}")
            raise error

    @staticmethod
    def recover_stale_jobs(db: Session) -> int:
        """
        Identify and mark orphaned or active processing jobs from previous server runs as failed.

        Executed during startup lifecycle to restore deterministic system state.

        Args:
            db (Session): Active database session.

        Returns:
            int: Number of orphaned jobs transitioned to failed.
        """
        logging.info("Executing JobCRUD.recover_stale_jobs")
        try:
            stmt = (
                update(JobModel)
                .where(JobModel.status.in_(["processing", "queued"]))
                .values(
                    status="failed",
                    error_code="SERVER_RESTARTED",
                    error_message="The server was restarted while this job was pending or in progress.",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            result = db.execute(stmt)
            db.commit()
            recovered_count = result.rowcount or 0
            if recovered_count > 0:
                logging.warning(f"Recovered and marked {recovered_count} stale job(s) as failed")
            return recovered_count
        except Exception as error:
            db.rollback()
            logging.error(f"Error in JobCRUD.recover_stale_jobs: {error}")
            raise error

    @staticmethod
    def get_expired_jobs(db: Session, reference_time: datetime) -> List[JobModel]:
        """
        Query all jobs whose expiration threshold has passed and are not yet marked expired.

        Args:
            db (Session): Active database session.
            reference_time (datetime): Current UTC timestamp to compare against expires_at.

        Returns:
            List[JobModel]: List of jobs requiring artifact cleanup.
        """
        logging.info("Executing JobCRUD.get_expired_jobs")
        try:
            query = select(JobModel).where(
                JobModel.expires_at <= reference_time,
                JobModel.status != "expired",
            )
            return list(db.execute(query).scalars().all())
        except Exception as error:
            logging.error(f"Error in JobCRUD.get_expired_jobs: {error}")
            raise error

    @staticmethod
    def delete_job(db: Session, job_id: str) -> bool:
        """
        Remove a job record permanently from the persistence store.

        Args:
            db (Session): Active database session.
            job_id (str): Unique identifier of target job.

        Returns:
            bool: True if record was found and deleted, False otherwise.
        """
        logging.info(f"Executing JobCRUD.delete_job for job '{job_id}'")
        try:
            job = db.execute(select(JobModel).where(JobModel.id == job_id)).scalar_one_or_none()
            if not job:
                return False
            db.delete(job)
            db.commit()
            return True
        except Exception as error:
            db.rollback()
            logging.error(f"Error in JobCRUD.delete_job for job '{job_id}': {error}")
            raise error

    @staticmethod
    def count_active_jobs_for_client(db: Session, client_key: str) -> int:
        """
        Count active (queued or processing) jobs associated with a client (hashed IP).

        Args:
            db (Session): Active database session.
            client_key (str): Client (hashed IP).

        Returns:
            int: Number of currently active jobs.
        """
        logging.info("Executing JobCRUD.count_active_jobs_for_client")
        try:
            query = select(JobModel).where(
                JobModel.client_key == client_key,
                JobModel.status.in_(["queued", "processing"]),
            )
            return len(list(db.execute(query).scalars().all()))
        except Exception as error:
            logging.error(f"Error in JobCRUD.count_active_jobs_for_client: {error}")
            return 0

    @staticmethod
    def update_chunk_progress(
        db: Session,
        job_id: str,
        chunk_done: int,
        chunk_total: int,
        eta_seconds: Optional[int] = None,
    ) -> None:
        """
        Persist chunk pipeline progress without changing job status.

        Called after each successful chunk synthesis so the frontend can display
        real-time progress (chunk x of y, time left).

        Args:
            db (Session): Active database session.
            job_id (str): Unique job identifier.
            chunk_done (int): Number of chunks finished so far.
            chunk_total (int): Total chunks planned for this job.
            eta_seconds (Optional[int]): Rolling ETA in seconds; None if not yet calculable.

        Returns:
            None

        Raises:
            Exception: If the database write fails.
        """
        logging.info(
            "Executing JobCRUD.update_chunk_progress for job '%s': %d/%d (eta=%s)",
            job_id, chunk_done, chunk_total, eta_seconds,
        )
        try:
            stmt = (
                update(JobModel)
                .where(JobModel.id == job_id)
                .values(
                    chunk_done=chunk_done,
                    chunk_total=chunk_total,
                    eta_seconds=eta_seconds,
                    updated_at=datetime.now(timezone.utc),
                    progress_message=(
                        f"Synthesising chunk {chunk_done} of {chunk_total}"
                        + (f" — ETA {eta_seconds}s" if eta_seconds is not None else "")
                    ),
                )
            )
            db.execute(stmt)
            db.commit()
        except Exception as error:
            db.rollback()
            logging.error(f"Error in JobCRUD.update_chunk_progress for job '{job_id}': {error}")
            raise error

    @staticmethod
    def get_daily_generated_minutes(db: Session, client_key: str) -> float:
        """
        Sum the total generated audio minutes for a client (hashed IP) in the last 24 hours.

        Reads from output_metadata.duration_seconds of completed jobs within the
        rolling 24-hour window to enforce CLIENT_DAILY_MINUTES quota.

        Args:
            db (Session): Active database session.
            client_key (str): Client (hashed IP).

        Returns:
            float: Total generated minutes in the last 24 hours.
        """
        logging.info("Executing JobCRUD.get_daily_generated_minutes for client (hashed IP)")
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            query = select(JobModel).where(
                JobModel.client_key == client_key,
                JobModel.status == "completed",
                JobModel.completed_at >= cutoff,
            )
            jobs = list(db.execute(query).scalars().all())
            total_seconds = sum(
                (j.output_metadata or {}).get("duration_seconds", 0.0)
                for j in jobs
            )
            return total_seconds / 60.0
        except Exception as error:
            logging.error(f"Error in JobCRUD.get_daily_generated_minutes: {error}")
            return 0.0
