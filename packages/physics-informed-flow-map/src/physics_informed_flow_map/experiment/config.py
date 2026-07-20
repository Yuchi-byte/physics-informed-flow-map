"""Typed experiment configuration: a pydantic schema validated from Hydra output.

The Config class supplements hydra (that assembles a raw config dict from files + CLI).

Then the values are passed to Config, which only validates rather than overriding any configuration parameters.
CLI overrides always has the highest priority, followed by the yaml files. The default values in the Config class are the lowest priority.
A ``run.py`` entry point composes a Hydra ``DictConfig`` from its ``conf/``
tree, then calls :meth:`Config.from_dictconfig` to validate it into the schema.
"""

from __future__ import annotations

from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="Config")


class Config(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )  # so that unknown keys (eg. typo'd override) are rejected and fails loudly.

    @classmethod
    def from_dictconfig(cls: type[T], cfg: DictConfig) -> T:
        """Validate a Hydra-composed ``DictConfig`` into this typed schema."""
        container = OmegaConf.to_container(cfg, resolve=True)
        return cls.model_validate(container)

    def dump(self) -> dict[str, Any]:
        """JSON-ready dict of the resolved config, pinned into the wandb run."""
        return self.model_dump(mode="json")
