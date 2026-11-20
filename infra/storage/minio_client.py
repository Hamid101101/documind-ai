
import hashlib
import logging
import os
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    MinIO object storage client for DocuMind AI.

    Handles:
        - Bucket connectivity checks
        - File uploads
        - File downloads
        - File deletion
        - Object existence checks
    """

    def __init__(self) -> None:
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")

        if not endpoint.startswith(("http://", "https://")):
            secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
            scheme = "https" if secure else "http"
            endpoint = f"{scheme}://{endpoint}"

        self.bucket = os.getenv(
            "MINIO_BUCKET",
            "documind-documents",
        )

        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")

        if not access_key or not secret_key:
            raise ValueError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY "
                "must be configured in the environment."
            )

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.getenv("MINIO_REGION", "us-east-1"),
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=5,
                read_timeout=30,
                retries={
                    "max_attempts": 3,
                    "mode": "standard",
                },
            ),
        )

        logger.info("MinIO client initialized.")

    @staticmethod
    def _validate_object_key(object_key: str) -> None:
        """Validate an object key before performing storage operations."""
        if (
            not object_key
            or object_key.startswith("/")
            or "\x00" in object_key
        ):
            raise ValueError("Invalid MinIO object key.")

    def check_connection(self) -> bool:
        """Check connectivity and access to the configured bucket."""
        try:
            self.client.head_bucket(Bucket=self.bucket)

            logger.info(
                "Successfully connected to MinIO bucket '%s'.",
                self.bucket,
            )
            return True

        except (ClientError, BotoCoreError):
            logger.exception("MinIO connection check failed.")
            return False

    def upload_file(
        self,
        file_obj: BinaryIO,
        object_key: str,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload a binary file object to MinIO.

        Returns:
            The stored object key.
        """
        self._validate_object_key(object_key)

        extra_args = {
            "ContentType": content_type,
        }

        if metadata:
            extra_args["Metadata"] = metadata

        try:
            self.client.upload_fileobj(
                Fileobj=file_obj,
                Bucket=self.bucket,
                Key=object_key,
                ExtraArgs=extra_args,
            )

            logger.info("Successfully uploaded object '%s'.", object_key)

            return object_key

        except (ClientError, BotoCoreError):
            logger.exception("Failed to upload object '%s'.", object_key)
            raise

    def download_file(
        self,
        object_key: str,
        destination: str | Path,
    ) -> Path:
        """Download an object from MinIO to a local file."""
        self._validate_object_key(object_key)

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.download_file(
                Bucket=self.bucket,
                Key=object_key,
                Filename=str(destination),
            )

            logger.info("Successfully downloaded '%s'.", object_key)

            return destination

        except (ClientError, BotoCoreError):
            logger.exception("Failed to download '%s'.", object_key)
            raise

    def get_object_stream(self, object_key: str):
        """Retrieve an object as a streaming response from MinIO."""
        self._validate_object_key(object_key)

        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=object_key,
            )

            logger.info("Opened streaming response for '%s'.", object_key)

            return response

        except (ClientError, BotoCoreError):
            logger.exception(
                "Failed to open streaming response for '%s'.",
                object_key,
            )
            raise

    def delete_file(self, object_key: str) -> None:
        """Delete an object from MinIO."""
        self._validate_object_key(object_key)

        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=object_key,
            )

            logger.info("Successfully deleted '%s'.", object_key)

        except (ClientError, BotoCoreError):
            logger.exception("Failed to delete '%s'.", object_key)
            raise

    def object_exists(self, object_key: str) -> bool:
        """Check whether an object exists in the configured bucket."""
        self._validate_object_key(object_key)

        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=object_key,
            )
            return True

        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")

            if error_code in ("404", "NoSuchKey", "NotFound"):
                return False

            logger.exception(
                "Failed to check object '%s'.",
                object_key,
            )
            raise

        except BotoCoreError:
            logger.exception(
                "MinIO communication error checking '%s'.",
                object_key,
            )
            raise

    
    def upload_multipart_file(
        self,
        file_obj: BinaryIO,
        object_key: str,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
        read_chunk_size: int = 1024 * 1024,
        part_size: int = 5 * 1024 * 1024,
    ) -> dict:
        """
        Upload a file using S3 multipart upload.

        Reads the source progressively, calculates SHA-256,
        and uploads parts without loading the entire file into memory.

        Returns:
            {
                "object_key": str,
                "size": int,
                "checksum": str,
            }
        """

        self._validate_object_key(object_key)

        if read_chunk_size <= 0:
            raise ValueError("read_chunk_size must be positive.")

        if part_size < 5 * 1024 * 1024:
            raise ValueError("S3 multipart parts must be at least 5 MiB.")

        if read_chunk_size > part_size:
            raise ValueError(
                "read_chunk_size cannot exceed part_size."
            )

        upload_id = None

        file_size = 0
        part_number = 1
        parts = []

        checksum = hashlib.sha256()
        buffer = bytearray()

        extra_args = {
            "Bucket": self.bucket,
            "Key": object_key,
            "ContentType": content_type,
        }

        if metadata:
            extra_args["Metadata"] = metadata

        try:
            # 1. Initiate multipart upload
            response = self.client.create_multipart_upload(
                **extra_args
            )

            upload_id = response["UploadId"]

            logger.info(
                "Started multipart upload for '%s'.",
                object_key,
            )

            # 2. Read and upload progressively
            while True:
                chunk = file_obj.read(read_chunk_size)

                if not chunk:
                    break

                if not isinstance(chunk, bytes):
                    raise TypeError(
                        "file_obj.read() must return bytes."
                    )

                checksum.update(chunk)
                file_size += len(chunk)
                buffer.extend(chunk)

                # Upload complete parts
                while len(buffer) >= part_size:
                    part_data = bytes(buffer[:part_size])
                    del buffer[:part_size]

                    part_response = self.client.upload_part(
                        Bucket=self.bucket,
                        Key=object_key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=part_data,
                    )

                    parts.append({
                        "PartNumber": part_number,
                        "ETag": part_response["ETag"],
                    })

                    logger.debug(
                        "Uploaded part %s for '%s'.",
                        part_number,
                        object_key,
                    )

                    part_number += 1

            # 3. Reject empty files
            if file_size == 0:
                raise ValueError("Cannot upload an empty file.")

            # 4. Upload remaining bytes (final part)
            if buffer:
                part_response = self.client.upload_part(
                    Bucket=self.bucket,
                    Key=object_key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=bytes(buffer),
                )

                parts.append({
                    "PartNumber": part_number,
                    "ETag": part_response["ETag"],
                })

            # 5. Complete multipart upload
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=object_key,
                UploadId=upload_id,
                MultipartUpload={
                    "Parts": parts,
                },
            )

            sha256_checksum = checksum.hexdigest()

            logger.info(
                "Successfully uploaded '%s' (%s bytes).",
                object_key,
                file_size,
            )

            return {
                "object_key": object_key,
                "size": file_size,
                "checksum": sha256_checksum,
            }

        except Exception:
            logger.exception(
                "Multipart upload failed for '%s'.",
                object_key,
            )

            # 6. Abort incomplete multipart upload
            if upload_id:
                try:
                    self.client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=object_key,
                        UploadId=upload_id,
                    )

                    logger.info(
                        "Aborted multipart upload for '%s'.",
                        object_key,
                    )

                except (ClientError, BotoCoreError):
                    logger.exception(
                        "Failed to abort multipart upload for '%s'.",
                        object_key,
                    )

            raise

