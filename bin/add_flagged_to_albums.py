"""
Add photos with the flags to static albums, so I can work with them on my phone.
"""
from osxphotos import PhotosDB, PhotosAlbum

from lib.osxphotos_utils import construct_query_options


def add_flagged_to_albums():
    # Open the Photos library
    photosdb = PhotosDB()

    # Define a mapping of album names to keywords (multiple keywords per album)
    album_keyword_map = {
        "Flagged: Deletion": ["flagged_qr"],
        "Flagged: 270º Rotation": ["flagged_rotated_90"],
        "Flagged: 180º Rotation": ["flagged_rotated_180"],
        "Flagged: 90º Rotation": ["flagged_rotated_270"],
        "Flagged: Documents": ["flagged_document_handwritten", "flagged_document_presentation"],
        "Flagged: NSFW": ["flagged_nsfw"],
        "Managed: Memes": ["flagged_meme"],
    }

    # Iterate over the dictionary to call add_to_album
    for album_name, keywords in album_keyword_map.items():
        for keyword in keywords:
            add_to_album(photosdb, keyword=keyword, album_name=album_name)


def add_to_album(photosdb, keyword, album_name):
    """
    Creates all these albums under the Utils folder
    :param photosdb:
    :param keyword:
    :param album_name:
    :return:
    """
    query_options = construct_query_options(keywords=[keyword], exclude_keywords=[f"validated_{keyword}"])
    photos = photosdb.query(query_options)

    if not photos or len(photos) == 0:
        print(f"No photos found with keyword '{keyword}'.")
        return

    print(f"Found {len(photos)} photos with keyword '{keyword}'.")

    album = PhotosAlbum(name=f"Utils/{album_name}", split_folder="/")
    for photo in photos:
        try:
            album.add(photo)
        except Exception as e:
            print(f"Error adding photo to album: {e}")

    print(f"Added {len(photos)} photos to album '{album_name}'.")


if __name__ == "__main__":
    add_flagged_to_albums()
