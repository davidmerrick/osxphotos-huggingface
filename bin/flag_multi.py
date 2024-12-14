"""
Runs multiple classifiers on photos simultaneously
"""

import click

from lib.classify.meme import MemeClassifier
from lib.classify.qr import QRClassifier
from lib.classify.screenshot import ScreenshotClassifier
from lib.common_options import common_options
from lib.photoflagger import PhotoFlagger


@click.command()
@common_options
def flag_photos(verbose_mode, dry_run, reset, library_path, selected, confidence_threshold):
    classifiers = [
        ScreenshotClassifier(confidence_threshold=confidence_threshold, enabled=False),
        MemeClassifier(confidence_threshold=confidence_threshold),
        QRClassifier(confidence_threshold=confidence_threshold)
    ]

    enabled_classifiers = [classifier for classifier in classifiers if classifier.enabled]

    PhotoFlagger(
        verbose_mode=verbose_mode,
        library_path=library_path,
        classifier=enabled_classifiers,
        keystore_name="flag_multi.db"
    ).flag_photos(dry_run=dry_run,reset=reset,selected=selected)


if __name__ == "__main__":
    flag_photos()
