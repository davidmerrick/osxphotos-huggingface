"""
Trains a classifier to recognize memes vs non-memes.

To use, put your meme photos in a library called "Training: Memes", and your control group
in an album called "Training: Not Memes." Note that the labels are an array, so you can train a model
on any number of albums that you want.
"""

import click

from lib.common_options import dry_run, library_path, verbose_mode
from lib.train import ModelTuner


@click.command()
@dry_run
@library_path
@verbose_mode
def train_memes(dry_run, library_path, verbose_mode):
    label_album_mapping = [
        ("meme", "Training: Not Memes"),
        ("non-meme", "Training: Memes"),
    ]

    trainer = ModelTuner(verbose_mode=verbose_mode, library_path=library_path)
    trainer.train(label_album_mapping, dry_run=dry_run, epochs=10)


if __name__ == "__main__":
    train_memes()
