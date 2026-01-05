"""UCAC4 catalog access using local zone files."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

_UCAC4_DTYPE = np.dtype(
    [
        ("pad", "u1", 4),
        ("ra", "<u4"),
        ("dec", "<i4"),
        ("rest", "u1", 68),
    ]
)
_RECORD_SIZE = 80
_MAS_TO_DEG = 1.0 / (1000.0 * 3600.0)


@dataclass
class Ucac4QueryResult:
    ra_deg: np.ndarray
    dec_deg: np.ndarray


class Ucac4Index:
    """Memory-mapped UCAC4 zone reader with RA-sorted queries."""

    def __init__(self, root: Path, cache_size: int = 4) -> None:
        self.root = Path(root)
        self.cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[int, np.memmap]" = OrderedDict()

    def _zone_path(self, zone: int) -> Path:
        return self.root / f"Z{zone:03d}.UC4"

    def _load_zone(self, zone: int) -> np.memmap | None:
        if zone in self._cache:
            self._cache.move_to_end(zone)
            return self._cache[zone]

        path = self._zone_path(zone)
        if not path.exists():
            return None

        file_size = path.stat().st_size
        data_size = file_size
        if data_size <= 0 or data_size % _RECORD_SIZE != 0:
            raise ValueError(
                "UCAC4 zone file size is not a multiple of the record size "
                f"({_RECORD_SIZE} bytes): "
                f"{path} ({file_size} bytes). "
                "This typically indicates the wrong catalog directory or a "
                "corrupted zone file."
            )

        mm = np.memmap(path, dtype=_UCAC4_DTYPE, mode="r")
        if mm.size == 0:
            return None

        self._cache[zone] = mm
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return mm

    def query_box(
        self,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
        *,
        thin: int = 1,
    ) -> Ucac4QueryResult:
        """Query catalog stars inside a RA/Dec rectangle in degrees."""

        thin = max(1, int(thin))
        ra_min_deg %= 360.0
        ra_max_deg %= 360.0
        dec_min_deg = max(-90.0, dec_min_deg)
        dec_max_deg = min(90.0, dec_max_deg)

        zones = _zones_for_dec_range(dec_min_deg, dec_max_deg)
        intervals = _ra_intervals(ra_min_deg, ra_max_deg)

        ra_out: list[np.ndarray] = []
        dec_out: list[np.ndarray] = []
        dec_min_mas = dec_min_deg / _MAS_TO_DEG
        dec_max_mas = dec_max_deg / _MAS_TO_DEG

        for zone in zones:
            mm = self._load_zone(zone)
            if mm is None:
                continue
            ra_mas = mm["ra"]
            dec_mas = mm["dec"]
            for ra_start_deg, ra_end_deg in intervals:
                ra_start_mas = ra_start_deg / _MAS_TO_DEG
                ra_end_mas = ra_end_deg / _MAS_TO_DEG
                lo = np.searchsorted(ra_mas, ra_start_mas, side="left")
                hi = np.searchsorted(ra_mas, ra_end_mas, side="right")
                if hi <= lo:
                    continue
                ra_slice = ra_mas[lo:hi:thin]
                dec_slice = dec_mas[lo:hi:thin]
                mask = (dec_slice >= dec_min_mas) & (dec_slice <= dec_max_mas)
                if not np.any(mask):
                    continue
                ra_out.append(ra_slice[mask].astype(np.float64) * _MAS_TO_DEG)
                dec_out.append(dec_slice[mask].astype(np.float64) * _MAS_TO_DEG)

        if not ra_out:
            return Ucac4QueryResult(np.empty(0), np.empty(0))

        return Ucac4QueryResult(np.concatenate(ra_out), np.concatenate(dec_out))


def _zones_for_dec_range(dec_min_deg: float, dec_max_deg: float) -> Iterable[int]:
    zone_min = int(np.floor(dec_min_deg + 90.0))
    zone_max = int(np.floor(dec_max_deg + 90.0))
    zone_min = max(0, min(179, zone_min))
    zone_max = max(0, min(179, zone_max))
    return range(zone_min, zone_max + 1)


def _ra_intervals(ra_min_deg: float, ra_max_deg: float) -> Tuple[Tuple[float, float], ...]:
    if ra_min_deg <= ra_max_deg:
        return ((ra_min_deg, ra_max_deg),)
    return ((ra_min_deg, 360.0), (0.0, ra_max_deg))
