"""Typed experiment configuration: a pydantic schema with OmegaConf override merge.

Each experiment framework subclasses :class:`Config`, declaring its knobs as typed
fields (nest further :class:`Config` subclasses for grouped knobs). ``resolve``
composes the field defaults with an optional variant preset and ``key=value`` CLI
overrides, then validates the result. Unknown keys are rejected (``extra="forbid"``),
so a typo'd override fails loudly instead of being silently ignored.
"""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base for experiment configs. Validates strictly; serialises round-trippably."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def resolve(
        cls,
        variant: dict[str, Any] | None = None,
        overrides: list[str] | None = None,
    ) -> "Config":
        """Compose ``defaults <- variant <- overrides`` and validate into the schema.

        ``variant`` is a preset dict (e.g. an entry from a framework's ``VARIANTS``);
        ``overrides`` is a dotlist of ``key=value`` / ``a.b=value`` strings (typically
        ``sys.argv[1:]``). Later sources win.
        """
        merged = OmegaConf.create(cls().model_dump())
        if variant:
            merged = cast(
                DictConfig, OmegaConf.merge(merged, OmegaConf.create(variant))
            )
        if overrides:
            merged = cast(
                DictConfig,
                OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides))),
            )
        container = OmegaConf.to_container(merged, resolve=True)
        return cls.model_validate(container)

    def dump(self) -> dict[str, Any]:
        """JSON-ready dict of the resolved config, pinned into the run manifest."""
        return self.model_dump(mode="json")
