"""
Trains a classifier to recognize different classes of things.
"""

import click

from lib.common_options import library_path, verbose_mode, config_path
from lib.config import parse_training_config
from lib.train import ModelTuner


@click.command()
@library_path
@verbose_mode
@config_path
def train_models(library_path, verbose_mode, config_path):
    configs = parse_training_config(config_path)
    for config in configs:
        trainer = ModelTuner(
            verbose_mode=verbose_mode,
            library_path=library_path,
            base_model=config.base_model,
            output_path=config.output_path
        )
        trainer.train(config.label_album_mapping, epochs=config.epochs)


if __name__ == "__main__":
    train_models()
