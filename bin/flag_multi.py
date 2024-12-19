"""
Runs multiple classifiers on photos simultaneously
"""

import click

from lib.classify.barcode import BarcodeClassifier
from lib.classify.meme import MemeClassifier
from lib.classify.qr import QRClassifier
from lib.classify.rotation import RotatedClassifier
from lib.common_options import common_options, env
from lib.photoflagger import PhotoFlagger


@click.command()
@common_options
@env
def flag_photos(
    verbose_mode,
    dry_run,
    reset,
    library_path,
    selected,
    confidence_threshold,
    env
):
    classifiers = [
        MemeClassifier(confidence_threshold=confidence_threshold),
        QRClassifier(confidence_threshold=confidence_threshold),
        BarcodeClassifier(confidence_threshold=confidence_threshold),
        RotatedClassifier(confidence_threshold=confidence_threshold)
    ]

    enabled_classifiers = [classifier for classifier in classifiers if classifier.enabled]

    PhotoFlagger(
        verbose_mode=verbose_mode,
        library_path=library_path,
        classifiers=enabled_classifiers,
        keystore_name=f"{env}_flag_multi.db"
    ).process_photos(
        dry_run=dry_run,
        reset=reset,
        selected=selected
    )


if __name__ == "__main__":
    flag_photos()
