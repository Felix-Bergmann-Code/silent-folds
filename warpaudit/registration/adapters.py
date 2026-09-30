"""Discovery boundary for isolated upstream registration adapters."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path

from ..config import PipelineConfig
from ..types import Registrar

__all__ = ["AdapterUnavailable", "load_registrar", "register_adapter"]


class AdapterUnavailable(LookupError):
    """No verified adapter entry point is installed for a configured pipeline."""


Factory = Callable[[PipelineConfig], Registrar]
_FACTORIES: dict[str, Factory] = {}


def register_adapter(name: str, factory: Factory) -> None:
    if not name:
        raise ValueError("adapter name cannot be empty")
    _FACTORIES[name] = factory


def _entrypoint_factories() -> dict[str, Factory]:
    discovered: dict[str, Factory] = {}
    selected = entry_points().select(group="warpaudit.registrars")
    for entrypoint in selected:
        discovered[entrypoint.name] = entrypoint.load()
    return discovered


def load_registrar(pipeline: PipelineConfig, *, project_root: Path | None = None) -> Registrar:
    """Instantiate a registered/entry-point adapter for one pipeline config.

    Entry points use group ``warpaudit.registrars`` and may be named by either
    the pipeline id or matcher id.  The factory receives the validated frozen
    ``PipelineConfig`` and must return an object satisfying ``Registrar``.
    """
    if pipeline.adapter == "subprocess":
        from .subprocess_adapter import subprocess_factory

        return subprocess_factory(pipeline, project_root=project_root)

    factories = {**_entrypoint_factories(), **_FACTORIES}
    name = pipeline.id if pipeline.id in factories else pipeline.matcher
    if name not in factories:
        raise AdapterUnavailable(
            f"no adapter for pipeline {pipeline.id!r} (matcher {pipeline.matcher!r}); "
            "install its isolated package with a 'warpaudit.registrars' entry point"
        )
    registrar = factories[name](pipeline)
    if not isinstance(registrar, Registrar):
        raise TypeError(f"adapter factory {name!r} did not return a Registrar")
    if registrar.pipeline_id != pipeline.id:
        raise ValueError(
            f"adapter reports pipeline_id {registrar.pipeline_id!r}, expected {pipeline.id!r}"
        )
    return registrar
