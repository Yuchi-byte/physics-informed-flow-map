"""Typed experiment configuration: a pydantic schema validated from Hydra output.

Each experiment framework subclasses :class:`Config`, declaring its knobs as typed
fields. A ``run.py`` entry point composes a Hydra ``DictConfig`` from its ``conf/``
tree, then calls :meth:`Config.from_dictconfig` to validate it into the schema.
Unknown keys are rejected (``extra="forbid"``), so a typo'd override fails loudly
instead of being silently ignored.
"""

from __future__ import annotations

from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="Config")


class Config(BaseModel):
    """Base for experiment configs. Validates strictly; serialises round-trippably."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_dictconfig(cls: type[T], cfg: DictConfig) -> T:
        """Validate a Hydra-composed ``DictConfig`` into this typed schema.

        Resolves interpolations, converts to a plain container, then validates.
        ``extra="forbid"`` turns any key not declared on the subclass into a
        ``ValidationError``.
        """
        container = OmegaConf.to_container(cfg, resolve=True)
        return cls.model_validate(container)

    def dump(self) -> dict[str, Any]:
        """JSON-ready dict of the resolved config, pinned into the wandb run."""
        return self.model_dump(mode="json")
