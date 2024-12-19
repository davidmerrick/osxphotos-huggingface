import datetime
from dataclasses import dataclass, field
from typing import List, Optional

from osxphotos import QueryOptions, PhotosAlbum


@dataclass
class EnhancedQueryOptions:
    """
    Wraps the osxphotos.QueryOptions and adds support for excluding keywords
    """
    keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    exclude_extensions: List[str] = field(default_factory=list)
    include_extensions: List[str] = field(default_factory=list)
    person: List[str] = field(default_factory=list)
    album: Optional[str] = None
    favorite: Optional[bool] = None
    selected: Optional[bool] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None

    def to_query_options(self):
        exclude_keywords_sql = (
            " or ".join(f"'{kw.lower()}' in [k.lower() for k in photo.keywords]" for kw in self.exclude_keywords)
            if self.exclude_keywords
            else "False"
        )

        exclude_extensions_sql = (
            " or ".join(f"photo.filename.lower().endswith('.{extension.lower()}')" for extension in self.exclude_extensions)
            if self.exclude_extensions
            else "False"
        )

        include_extensions_sql = (
            " or ".join(f"photo.filename.lower().endswith('.{extension.lower()}')" for extension in self.include_extensions)
            if self.include_extensions
            else "True"
        )
        query_eval = f"not ({exclude_keywords_sql}) and not ({exclude_extensions_sql}) and ({include_extensions_sql})"

        return QueryOptions(
            movies=False,
            query_eval=[query_eval],
            **({"selected": self.selected} if self.selected else {}),
            **({"keyword": self.keywords} if self.keywords else {}),
            **({"album": self.album} if self.album else {}),
            **({"favorite": self.favorite} if self.favorite else {}),
            **({"person": self.person} if self.person else {}),
            **({"from_date": datetime.datetime.strptime(self.from_date, "%Y-%m-%d")} if self.from_date else {}),
            **({"to_date": datetime.datetime.strptime(self.to_date, "%Y-%m-%d")} if self.to_date else {})
        )


def construct_query_options(
    selected=None,
    keywords=None,
    exclude_keywords=[],
    album=None,
    favorite=None,
    person=None
):
    return EnhancedQueryOptions(
        selected=selected,
        keywords=keywords,
        exclude_keywords=exclude_keywords,
        album=album,
        favorite=favorite,
        person=person
    )


def add_to_album(photos, album_name, prefix="Utils"):
    """
    Adds all photos to an album under "prefix/album_name"
    :param photosdb:
    :param album_name:
    :return:
    """
    album = PhotosAlbum(name=f"{prefix}/{album_name}", split_folder="/")
    for photo in photos:
        try:
            album.add(photo)
        except Exception as e:
            print(f"Error adding photo to album: {e}")

    print(f"Added {len(photos)} photos to album '{album_name}'.")
