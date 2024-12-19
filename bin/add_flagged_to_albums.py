"""
Add photos with the flags to static albums, so I can work with them on my phone.
"""

import click
from osxphotos import PhotosDB

from lib.common_options import config_path
from lib.config import parse_managed_albums
from lib.osxphotos_utils import add_to_album


@click.command()
@config_path
def add_flagged_to_albums(config_path):
    photosdb = PhotosDB()
    managed_albums = parse_managed_albums(config_path)

    # Iterate over the dictionary to call add_to_album
    for managed_album in managed_albums:
        print(f"Updating album {managed_album.name}")
        photos = set()  # Use a set to automatically handle duplicates
        for enhanced_query_options in managed_album.query_options:
            # Update the set with new photos from the query
            photos.update(photosdb.query(enhanced_query_options.to_query_options()))

        add_to_album(
            list(photos),  # Convert the set back to a list if `add_to_album` requires a list
            album_name=managed_album.name,
            prefix=managed_album.prefix
        )


if __name__ == "__main__":
    add_flagged_to_albums()
