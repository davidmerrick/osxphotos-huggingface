import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from osxphotos import QueryOptions
from photoscript import Photo


def add_keyword(photo, keyword: str):
    """
    Add a keyword to a photo.

    Args:
        photo (osxphotos.PhotoInfo): Photo to add keyword to
        keyword (str): Keyword to add
    """
    add_keywords(photo=photo, keywords=[keyword])


def add_keywords(photo, keywords):
    """
    Add multiple keywords to a photo.

    Args:
        photo (osxphotos.PhotoInfo): Photo to add keyword to
        keywords (List[str]): Keywords to add
    """
    photo_ = Photo(photo.uuid)
    photo_.keywords = list(set(photo_.keywords + keywords))

def construct_query_options(selected=None, keywords=None, exclude_keywords=[], album=None):
    """
    Construct query options to filter photos that might need rotation.
    The query options exclude photos with any of the specified 'exclude_keywords'.

    :param selected: If set, filters to only selected photos.
    :param exclude_keywords: A list of keywords to exclude. Default is an empty list.
    """
    exclude_logic = (
        " or ".join(f"'{kw.lower()}' in [k.lower() for k in photo.keywords]" for kw in exclude_keywords)
        if exclude_keywords
        else "False"
    )
    query_eval = f"not ({exclude_logic})"

    return QueryOptions(
        movies=False,
        album=album,
        query_eval=[query_eval],
        **({"selected": selected} if selected else {}),  # Include 'selected' only if it's not None
        **({"keyword": keywords} if keywords else {}),  # Include 'selected' only if it's not None
        **({"album": album} if album else {})  # Include 'selected' only if it's not None
    )

def validate_library_path(library_path):
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
