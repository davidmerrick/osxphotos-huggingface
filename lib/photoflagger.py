import json
import logging
import os.path
import sys
from enum import Enum
from typing import List

from loguru import logger
from osxphotos import PhotosDB
from osxphotos.cli.common import get_data_dir
from osxphotos.sqlitekvstore import SQLiteKVStore
from photoscript import Photo
from rich.console import Console
from rich.progress import Progress

from lib.classify import Classifier
from lib.osxphotos_utils import *

logger = logging.getLogger("photoflagger")


class ProcessResultStatus(Enum):
    SKIPPED = "skipped"
    ALREADY_PROCESSED = "already_processed"
    FLAGGED = "success"
    ERROR = "error"


@dataclass
class ProcessResult:
    status: ProcessResultStatus
    add_keywords: List[str] = field(default_factory=list)

    @classmethod
    def skipped(cls) -> "ProcessResult":
        return cls(status=ProcessResultStatus.SKIPPED)


@dataclass
class PhotoProcessContext:
    photo: Photo
    preview_path: str
    logger: object
    dry_run: bool


class PhotoFlagger:
    """
    A class to process photos from an Apple Photos library using one or more pretrained classifiers.
    """

    def __init__(
        self,
        keystore_name,
        library_path,
        classifiers: list[Classifier] = [],
        verbose_mode=False
    ):
        # Configure logging first
        self._console = Console(stderr=True)

        self._keystore_name = keystore_name
        self._console = Console(stderr=True)
        self._kvstore = self._get_kv_store()

        self._validate_library_path(library_path)
        self.photosdb = PhotosDB(dbfile=library_path)
        self.classifiers = classifiers
        self._configure_logging(verbose_mode)

    def _configure_logging(self, verbose_mode):
        """
        Configure the class-specific logger.
        """
        if verbose_mode:
            logger.setLevel(logging.DEBUG)

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

    def get_preview_paths(self, query_options: EnhancedQueryOptions):
        """
        Returns the preview paths for all photos matching the query.
        :param query_options:
        :return:
        """
        photos = self.photosdb.query(query_options.to_query_options())
        return [self._build_context(photo, dry_run=True).preview_path for photo in photos]

    def _get_exclude_keywords(self):
        return [f"validated_{classifier.name}" for classifier in self.classifiers]

    def process_photos(
        self,
        dry_run: bool = False,
        reset: bool = False,  # Whether to reset the database of previously processed photos
        selected: bool = False  # Whether to operate only on selected photos
    ):
        """
        Process a list of photos using the provided function.

        :param selected:
        :param reset:
        :param dry_run:
        """
        if reset:
            self._reset_kvstore()

        query_options = construct_query_options(selected, exclude_keywords=self._get_exclude_keywords()).to_query_options()

        # Track number of photos processed for reporting at the end
        photos = self.photosdb.query(query_options)
        num_photos = len(photos)
        num_previously_processed = 0
        num_skipped = 0
        num_error = 0
        num_flagged = 0

        with (Progress(console=self._console) as progress):
            task = progress.add_task(f"Processing {num_photos} photos", total=num_photos)
            for photo in photos:
                logger.debug(f"Processing photo: {photo.filename}")
                ctx = self._build_context(photo, dry_run)

                if photo.path is None or not os.path.exists(photo.path):
                    num_skipped += 1
                    logger.debug("File does not exist. Skipping.")
                    continue
                elif self._kvstore.get(photo.uuid):
                    logger.debug(f"Skipping previously processed photo {photo.original_filename} ({photo.uuid})")
                    num_previously_processed += 1
                    continue
                try:
                    result = self._process_photo(ctx)
                    if result.status == ProcessResultStatus.FLAGGED:
                        logger.debug(f"Flagged photo {photo.filename}")
                        if not dry_run and result.add_keywords:
                            self._add_keywords(photo, result.add_keywords)
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

    def _process_photo(self, ctx: PhotoProcessContext) -> ProcessResult:
        if not ctx.photo.path_derivatives:
            ctx.logger.debug(f"Skipping {ctx.original_filename}; could not find photo path")
            return ProcessResult(status=ProcessResultStatus.SKIPPED)

        # Use all classifiers to process the photo
        flags = []
        for classifier in self.classifiers:
            classification = classifier.classify(ctx.photo.path_derivatives[0])
            if classification:
                if isinstance(classification, bool):
                    flags.append(f"flagged_{classifier.name}")
                else:
                    flags.append(f"flagged_{classifier.name}_{classification}")

        if len(flags) > 0:
            ctx.logger.debug(f"Image flagged with keywords: {', '.join(flags)}")
            return ProcessResult(ProcessResultStatus.FLAGGED, flags)
        else:
            ctx.logger.debug("Image was not flagged")
            return ProcessResult(ProcessResultStatus.SKIPPED)

    def _add_keywords(self, photo, keywords):
        """
        Add multiple keywords to a photo.

        Args:
            photo (osxphotos.PhotoInfo): Photo to add keyword to
            keywords (List[str]): Keywords to add
        """
        photo_ = Photo(photo.uuid)
        photo_.keywords = list(set(photo_.keywords + keywords))

    def _validate_library_path(self, library_path):
        """
        Validates that the library path exists, is a directory, and ends with 'photoslibrary'.
        """
        if not os.path.exists(library_path):
            sys.exit(f"Error: The path '{library_path}' does not exist.")
        if not os.path.isdir(library_path):
            sys.exit(f"Error: The path '{library_path}' is not a directory.")
        if not library_path.endswith("photoslibrary"):
            sys.exit(f"Error: The path '{library_path}' must end with 'photoslibrary'.")
        return library_path
