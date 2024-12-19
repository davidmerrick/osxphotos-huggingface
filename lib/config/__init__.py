import os
from dataclasses import dataclass, field
from typing import List

import yaml

from lib.osxphotos_utils import EnhancedQueryOptions


@dataclass
class ManagedAlbum:
    name: str
    prefix: str
    query_options: List[EnhancedQueryOptions] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str
    output_path: str
    epochs: int = 5
    base_model: str = "google/vit-base-patch16-224"
    label_album_mapping: List[tuple] = field(default_factory=list)


def _get_config(config_path: str):
    sanitized_path = os.path.expanduser(config_path)
    with open(sanitized_path, 'r') as file:
        return yaml.safe_load(file)


def parse_managed_albums(config_path: str) -> List[ManagedAlbum]:
    data = _get_config(config_path)

    managed_albums_data = data.get('managed_albums', [])
    managed_albums = [
        ManagedAlbum(
            name=album.get('name'),
            prefix=album.get('prefix'),
            query_options=[EnhancedQueryOptions(**query_options) for query_options in (album.get('query_options') or [])]
        )
        for album in managed_albums_data
    ]
    return managed_albums


def parse_training_config(config_path: str) -> List[ModelConfig]:
    training_data = _get_config(config_path).get('training', [])
    return [ModelConfig(**config) for config in training_data]
