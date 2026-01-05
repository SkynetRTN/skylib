"""Atlas catalog registry for plate solving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .ucac4 import Ucac4Index
from .ucac5 import Ucac5Index


@dataclass(frozen=True)
class CatalogSpec:
    name: str
    index_factory: Callable[[Path], "CatalogIndex"]


class CatalogIndex(Protocol):
    def query_box(
        self,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
        *,
        thin: int = 1,
    ):
        ...


CATALOG_REGISTRY: Mapping[str, CatalogSpec] = {
    "ucac4": CatalogSpec(name="ucac4", index_factory=Ucac4Index),
    "ucac5": CatalogSpec(name="ucac5", index_factory=Ucac5Index),
}


def get_catalog_spec(name: str) -> CatalogSpec:
    normalized = name.strip().lower()
    if normalized not in CATALOG_REGISTRY:
        raise ValueError(f"Unsupported catalog: {name}")
    return CATALOG_REGISTRY[normalized]
