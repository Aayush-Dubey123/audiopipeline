"""
Database CRUD repository for voice catalog entities.

Provides transactional query, lookup, and idempotent synchronization for AI voice profiles.
"""

import json
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from core import logger
from core.models.voice_model import VoiceModel

logging = logger(__name__)


class VoiceCRUD:
    """
    Persistence operations for voice catalog records.
    """

    @staticmethod
    def get_voice_by_id(db: Session, voice_id: str) -> Optional[VoiceModel]:
        """
        Query a voice entity by its unique identifier.

        Args:
            db (Session): Active database session.
            voice_id (str): Unique identifier of the target voice.

        Returns:
            Optional[VoiceModel]: Persisted voice record or None if not found.
        """
        logging.info(f"Executing VoiceCRUD.get_voice_by_id for voice '{voice_id}'")
        try:
            query = select(VoiceModel).where(VoiceModel.id == voice_id)
            return db.execute(query).scalar_one_or_none()
        except Exception as error:
            logging.error(f"Error in VoiceCRUD.get_voice_by_id for voice '{voice_id}': {error}")
            raise error

    @staticmethod
    def list_voices(
        db: Session,
        model_type: Optional[str] = None,
        include_experimental: bool = False,
    ) -> List[VoiceModel]:
        """
        Retrieve catalog voices with optional filtering.

        Args:
            db (Session): Active database session.
            model_type (Optional[str]): Optional model type filter (e.g. chatterbox_reference).
            include_experimental (bool): Flag whether to include experimental status voices.

        Returns:
            List[VoiceModel]: Collection of matching voice models.
        """
        logging.info("Executing VoiceCRUD.list_voices")
        try:
            query = select(VoiceModel)
            if not include_experimental:
                query = query.where(VoiceModel.status != "experimental")
            if model_type:
                query = query.where(VoiceModel.model_type == model_type)
            query = query.order_by(VoiceModel.name.asc())
            return list(db.execute(query).scalars().all())
        except Exception as error:
            logging.error(f"Error in VoiceCRUD.list_voices: {error}")
            raise error

    @staticmethod
    def upsert_voice(db: Session, voice_data: dict) -> VoiceModel:
        """
        Insert or update a voice entry in the catalog database.

        Args:
            db (Session): Active database session.
            voice_data (dict): Dictionary of validated voice attributes.

        Returns:
            VoiceModel: Persisted or updated voice instance.
        """
        logging.info(f"Executing VoiceCRUD.upsert_voice for id '{voice_data.get('id')}'")
        try:
            voice_id = voice_data["id"]
            existing = db.execute(select(VoiceModel).where(VoiceModel.id == voice_id)).scalar_one_or_none()
            if existing:
                for key, val in voice_data.items():
                    setattr(existing, key, val)
                db.commit()
                db.refresh(existing)
                return existing
            new_voice = VoiceModel(**voice_data)
            db.add(new_voice)
            db.commit()
            db.refresh(new_voice)
            return new_voice
        except Exception as error:
            db.rollback()
            logging.error(f"Error in VoiceCRUD.upsert_voice: {error}")
            raise error

    @classmethod
    def sync_voices_from_json(cls, db: Session, json_path: Path, voices_base_dir: Path) -> int:
        """
        Idempotently load and synchronize voices from a JSON manifest.

        Skips invalid or missing entries with a warning rather than crashing.

        Args:
            db (Session): Active database session.
            json_path (Path): Path to the voices.json file.
            voices_base_dir (Path): Base directory for resolving voice assets.

        Returns:
            int: Number of valid voices successfully synchronized.
        """
        logging.info(f"Executing VoiceCRUD.sync_voices_from_json from {json_path}")
        if not json_path.is_file():
            logging.warning(f"Voice catalog manifest not found at {json_path}; skipping sync")
            return 0

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as parse_error:
            logging.warning(f"Failed to parse voice catalog manifest at {json_path}: {parse_error}")
            return 0

        if not isinstance(entries, list):
            logging.warning(f"Invalid catalog manifest format at {json_path}: expected JSON array")
            return 0

        synced_count = 0
        resolved_base = voices_base_dir.resolve()

        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                logging.warning(f"Skipping malformed non-dict voice entry: {raw_entry}")
                continue

            voice_id = raw_entry.get("id")
            if not voice_id or not isinstance(voice_id, str):
                logging.warning("Skipping voice entry missing required 'id' attribute")
                continue

            model_type = raw_entry.get("model_type")
            if model_type != "chatterbox_reference":
                logging.warning(f"Skipping voice '{voice_id}': invalid model_type '{model_type}'")
                continue

            asset_file = raw_entry.get("asset_file")
            if not asset_file or not isinstance(asset_file, str):
                logging.warning(f"Skipping voice '{voice_id}': missing 'asset_file'")
                continue

            # Path traversal and existence check
            asset_path = Path(asset_file)
            if asset_path.is_absolute() or ".." in asset_path.parts:
                logging.warning(f"Skipping voice '{voice_id}': illegal asset path traversal '{asset_file}'")
                continue

            full_asset = (resolved_base / asset_path).resolve()
            try:
                full_asset.relative_to(resolved_base)
            except ValueError:
                logging.warning(f"Skipping voice '{voice_id}': asset path escapes voices directory '{asset_file}'")
                continue

            if not full_asset.is_file():
                logging.warning(f"Skipping voice '{voice_id}': asset file not found at {full_asset}")
                continue

            if model_type in ("chatterbox_reference", "openvoice_reference") and full_asset.suffix.lower() not in (
                ".wav",
                ".ogg",
                ".flac",
                ".mp3",
            ):
                logging.warning(f"Skipping voice '{voice_id}': reference audio has unsupported extension '{full_asset.suffix}'")
                continue

            # Validate preview file if present
            preview_file = raw_entry.get("preview_file")
            validated_preview: Optional[str] = None
            if preview_file and isinstance(preview_file, str):
                preview_path = Path(preview_file)
                if not preview_path.is_absolute() and ".." not in preview_path.parts:
                    full_preview = (resolved_base / preview_path).resolve()
                    try:
                        full_preview.relative_to(resolved_base)
                        if full_preview.is_file():
                            validated_preview = preview_file
                        else:
                            logging.warning(f"Voice '{voice_id}' preview file not found on disk: {full_preview}")
                    except ValueError:
                        logging.warning(f"Voice '{voice_id}' preview file escapes voices directory: {preview_file}")

            voice_data = {
                "id": voice_id,
                "name": raw_entry.get("name", voice_id),
                "description": raw_entry.get("description"),
                "style": raw_entry.get("style"),
                "languages": raw_entry.get("languages", ["en"]),
                "model_type": model_type,
                "preview_file": validated_preview,
                "asset_file": asset_file,
                "license": raw_entry.get("license", "Unknown"),
                "usage_permission": raw_entry.get("usage_permission", "Operator restricted"),
                "status": raw_entry.get("status", "available"),
                "limitations": raw_entry.get("limitations"),
            }

            try:
                cls.upsert_voice(db, voice_data)
                synced_count += 1
            except Exception as upsert_err:
                logging.warning(f"Failed to upsert voice '{voice_id}': {upsert_err}")

        logging.info(f"Completed voice catalog synchronization: {synced_count} voice(s) loaded")
        return synced_count
