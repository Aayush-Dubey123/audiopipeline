"""
Database CRUD repository for inference job persistence entities.

Provides transactional job lifecycle updates, retrieval, stale job recovery, and cleanup queries.
"""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from core import logger
from core.models.job_model import JobModel

logging = logger(__name__)


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

            job.status = status
            job.updated_at = datetime.now(timezone.utc)
            if progress_message is not None:
                job.progress_message = progress_message
            if error_code is not None:
                job.error_code = error_code
            if error_message is not None:
                job.error_message = error_message
            if output_metadata is not None:
                job.output_metadata = output_metadata

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
