from __future__ import annotations

from contextlib import contextmanager
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Iterator


REGISTRATION_SINGLE_TEMPLATE = "registration_single_template.toml"
REGISTRATION_BATCH_TEMPLATE = "registration_batch_template.toml"
REGISTRATION_SWEEP_TEMPLATE = "registration_sweep_template.toml"


def get_template_resource(template_name: str):
    return importlib_resources.files(__name__).joinpath(template_name)


@contextmanager
def as_template_path(template_name: str) -> Iterator[Path]:
    with importlib_resources.as_file(get_template_resource(template_name)) as path:
        yield path


__all__ = [
    "REGISTRATION_BATCH_TEMPLATE",
    "REGISTRATION_SINGLE_TEMPLATE",
    "REGISTRATION_SWEEP_TEMPLATE",
    "as_template_path",
    "get_template_resource",
]
