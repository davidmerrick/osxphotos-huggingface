import datetime
import json
import os.path
from typing import Callable

from osxphotos import PhotosDB
from osxphotos.cli.common import get_data_dir
from osxphotos.sqlitekvstore import SQLiteKVStore
from rich.progress import Progress

from lib.osxphotos_utils import *

# Define the Callable type for process_photo
ProcessPhotoFunc = Callable[[PhotoProcessContext], ProcessResult]

import sys
from loguru import logger
from rich.console import Console
from rich.text import Text


def configure_global_logging(console, verbose_mode):
    """
    Configure the global logger to log through rich.Console.
    """
    logger.remove()  # Remove default loguru handler

    # Define a custom sink for rich.Console
    def rich_sink(message):
        log_record = message.record
        log_level = log_record["level"].name
        log_time = log_record["time"].strftime("%Y-%m-%d %H:%M:%S")
        log_message = log_record["message"]

        # Use rich.Text for styled output
        text = Text()
        text.append(f"[{log_time}] ", style="green")
        text.append(f"[{log_level}] ", style=f"{log_level.lower()}")
        text.append(f"{log_message}", style="white")
        console.print(text)

    # Add the custom sink to loguru
    logger.add(rich_sink, level="DEBUG" if verbose_mode else "INFO")

    if verbose_mode:
        logger.debug("Verbose mode enabled. Log level set to DEBUG.")
    else:
        logger.info("Log level set to INFO.")


class PhotoProcessor:
    """
    A class to process photos from an Apple Photos library using a custom function.

    Scope of responsibilities:
    * This handles ALL mutations to the library. No callers should do this.
    """

    def __init__(
        self,
        keystore_name,
        library_path,
        verbose_mode=False,
    ):
        # Configure logging first
        self._console = Console(stderr=True)
        configure_global_logging(self._console, verbose_mode)

        self._keystore_name = keystore_name
        self._console = Console(stderr=True)
        self._kvstore = self._get_kv_store()

        validate_library_path(library_path)
        self.photosdb = PhotosDB(dbfile=library_path)

    def _configure_logging(self, verbose_mode):
        """
        Configure the class-specific logger.
        """
        logger.remove()  # Remove default handler
        if verbose_mode:
            logger.add(sys.stdout, level="DEBUG", format="<green>{time}</green> - <level>{message}</level>")
            logger.debug("Verbose mode enabled. Log level set to DEBUG.")
        else:
            logger.add(sys.stdout, level="INFO", format="<green>{time}</green> - <level>{message}</level>")

    def _get_kv_store(self, reset=False):
        """
        Get the key-value store for storing processed photos.
        :param reset: Reset the database of previously processed photos.
        :return:
        """
        # Set the database path
        db_path = os.path.join(get_data_dir(), self._keystore_name)
        if reset:
            logger.debug(f"Resetting database: {db_path}")
            if os.path.exists(db_path):
                os.remove(db_path)
        logger.debug(f"Using database {db_path}")

        # enable write-ahead logging for performance, serialize/deserialize data as JSON
        return SQLiteKVStore(
            db_path,
            wal=True,
            serialize=json.dumps,
            deserialize=json.loads
        )

    def _reset_kvstore(self):
        self._kvstore = self._get_kv_store(reset=True)

    def _update_kvstore(self, photo):
        # record that will be stored in the kvstore database
        record = {
            "datetime": datetime.datetime.now().isoformat(),
            "uuid": photo.uuid
        }

        # store photo in kvstore to indicate it's been processed
        self._kvstore.set(photo.uuid, record)
        logger.debug(f"Stored photo {photo.uuid} in kvstore")

    def _build_context(self, photo, dry_run):
        """
        Build a ProcessContext object to pass to the process_photo function.
        """

        # Every photo has a JPEG preview image ("derivative" in Photos) so use that. The preview image is
        # smaller and lower-resolution, but should be good enough for most model detection
        preview_path = None
        if len(photo.path_derivatives) > 0:
            preview_path = photo.path_derivatives[0]

        return PhotoProcessContext(
            photo=photo,
            preview_path=preview_path,
            logger=logger,
            dry_run=dry_run
        )

    def get_contexts(self, query_options: QueryOptions, dry_run: bool):
        photos = self.photosdb.query(query_options)

        # Map the photos into contexts using self._build_context
        return [self._build_context(photo, dry_run) for photo in photos]

    def process_photos(
        self,
        query_options: QueryOptions,
        process_photo: ProcessPhotoFunc,
        dry_run: bool = False,
        reset: bool = False  # Whether to reset the database of previously processed photos
    ):
        """
        Process a list of photos using the provided function.

        :param reset:
        :param export:
        :param dry_run:
        :param query_options:
        :param process_photo: A function to process each photo, which returns a ProcessResult
        """
        if (reset):
            self._reset_kvstore()

        # Track number of photos processed for reporting at the end
        photo_ctxs = self.get_contexts(query_options, dry_run)
        num_photos = len(photo_ctxs)
        num_previously_processed = 0
        num_skipped = 0
        num_error = 0
        num_flagged = 0

        with (Progress(console=self._console) as progress):
            task = progress.add_task(f"Processing {num_photos} photos", total=num_photos)
            for ctx in photo_ctxs:
                photo = ctx.photo
                logger.debug(f"Processing photo: {photo.filename}")

                if photo.path is None or not os.path.exists(photo.path):
                    num_skipped += 1
                    logger.debug("File does not exist. Skipping.")
                    continue
                elif self._kvstore.get(photo.uuid):
                    logger.debug(f"Skipping previously processed photo {photo.original_filename} ({photo.uuid})")
                    num_previously_processed += 1
                    continue
                try:
                    result = process_photo(ctx)
                    if result.status == ProcessResultStatus.FLAGGED:
                        logger.debug(f"Flagged photo {photo.filename}")
                        if not dry_run and result.add_keywords:
                            add_keywords(photo, result.add_keywords)
                        num_flagged += 1
                    elif result.status == ProcessResultStatus.SKIPPED:
                        logger.debug(f"Skipped photo {photo.filename}")
                        num_skipped += 1
                    elif result.status == ProcessResultStatus.ERROR:
                        logger.debug(f"Errored on photo {photo.filename}")
                        num_error += 1
                except Exception as e:
                    logger.debug(f"Errored on photo {photo.filename}: {e}")
                    num_error += 1
                if not dry_run:
                    self._update_kvstore(photo)
                progress.advance(task)

        print(f"Processed {num_photos} photos")
        print(f"Previously processed {num_previously_processed} photos")
        print(f"Skipped {num_skipped} photos")
        print(f"Errored on {num_error} photos")
        print(f"Flagged {num_flagged} photos")
