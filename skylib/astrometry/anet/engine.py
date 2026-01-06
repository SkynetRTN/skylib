"""Astrometry.net solver engine wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

try:  # pragma: no cover - optional dependency
    from . import an_engine
except Exception:  # pragma: no cover - missing optional dependency
    an_engine = None


class AstrometryNetSolver:
    """Astrometry.net engine wrapper."""

    globs = None  # type: list

    def __init__(self, index_path: Union[str, Sequence[str]]):
        if an_engine is None:
            raise ImportError("an_engine module not available")

        if isinstance(index_path, str):
            index_path = [index_path]

        self.solver = an_engine.solver_new()

        self.indexes = []
        for path in index_path:
            for fn in Path(path).glob("*"):
                try:
                    idx = an_engine.index_load(str(fn), 0, None)
                    if idx is not None:
                        self.indexes.append(idx)
                except Exception:
                    pass

        if not self.indexes:
            raise ValueError("No indexes found")

        self.indexes.sort(key=lambda _idx: _idx.nquads)

        self.globs = []
        ngc_path = Path(__file__).with_name("ngc2000.dat")
        with ngc_path.open() as handle:
            for line in handle.read().splitlines():
                try:
                    typ = line[6:9].strip()
                    if typ != "Gb":
                        continue
                    ra_h, ra_m = line[10:12], line[13:17]
                    dec_s, dec_d, dec_m = line[19], line[20:22], line[23:25]
                    ra = int(ra_h) + float(ra_m) / 60
                    dec = (1 - 2 * (dec_s == "-")) * (
                        int(dec_d) + int(dec_m) / 60.0
                    )
                    r = float(line[33:38]) / 2
                    self.globs.append([ra, dec, r / 60])
                except Exception:
                    pass


__all__ = ["AstrometryNetSolver", "an_engine"]
