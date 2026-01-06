from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

_UCAC5_ZONE_RECORD_SIZE = 52  # u5z zone files (z001..z900)
_MAS_TO_DEG = 1.0 / (1000.0 * 3600.0)

# Define a fixed dtype for u5z:
_UCAC5_U5Z_DTYPE = np.dtype(
    [
        ("srcid", "<u8"),     # Gaia source id
        ("ra_mas", "<i4"),    # Gaia RA at epoch 2015.0, milliarcseconds
        ("dec_mas", "<i4"),   # Gaia Dec at epoch 2015.0, milliarcseconds
        ("rest", "u1", _UCAC5_ZONE_RECORD_SIZE - 16),
    ]
)

_UCAC5_ZONES = 900
_UCAC5_BINS_PER_ZONE = 1440
_UCAC5_ZONE_HEIGHT_DEG = 180.0 / _UCAC5_ZONES        # 0.2 deg
_UCAC5_BIN_WIDTH_DEG = 360.0 / _UCAC5_BINS_PER_ZONE  # 0.25 deg


@dataclass(frozen=True)
class Ucac5Span:
    """A contiguous record span in a z### file."""
    zone: int          # 1..900
    start: int         # 0-based record start index within zone file
    count: int         # number of records


@dataclass
class Ucac5QueryResult:
    ra_deg: np.ndarray
    dec_deg: np.ndarray


class Ucac5Index:
    """
    UCAC5 index reader for the standard u5z zone distribution.

    Expected on-disk layout:

      root/
        u5z/
          u5index.unf
          z001
          ...
          z900
    """

    def __init__(self, root: str | Path, *, u5z_subdir: str = "u5z", cache_size: int = 4):
        self.root = Path(root)
        self.u5z_dir = self.root / u5z_subdir
        self.cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[int, np.memmap]" = OrderedDict()

        if not self.u5z_dir.exists():
            raise FileNotFoundError(f"Missing UCAC5 u5z directory: {self.u5z_dir}")

        # Prefer ASCII index for correctness with this distribution
        self.index_asc = self.u5z_dir / "u5index.asc"
        if not self.index_asc.exists():
            raise FileNotFoundError(f"Missing UCAC5 ASCII index file: {self.index_asc}")

        # Allocate [zone, bin, (start,count)]
        self._idx = np.zeros((_UCAC5_ZONES, _UCAC5_BINS_PER_ZONE, 2), dtype=np.int32)

        # Parse u5index.asc: each line has start count zone bin (and optional dec on bin 1)
        # Example:
        # "     0   21 406    1  -8.8"
        # "   135141   18 406 1440"
        with self.index_asc.open("rt", encoding="ascii", errors="strict") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # start, count, zone, bin are always the first 4 tokens
                if len(parts) < 4:
                    continue
                start = int(parts[0])
                count = int(parts[1])
                zone = int(parts[2])
                bin_ = int(parts[3])
                if 1 <= zone <= _UCAC5_ZONES and 1 <= bin_ <= _UCAC5_BINS_PER_ZONE:
                    self._idx[zone - 1, bin_ - 1, 0] = start
                    self._idx[zone - 1, bin_ - 1, 1] = count

        starts = self._idx[:, :, 0].astype(np.int64)
        counts = self._idx[:, :, 1].astype(np.int64)
        self._zone_record_counts = np.max(starts + counts, axis=1).astype(int)

    def zone_path(self, zone: int) -> Path:
        if not (1 <= zone <= _UCAC5_ZONES):
            raise ValueError(f"zone out of range: {zone}")
        return self.u5z_dir / f"z{zone:03d}"

    def _load_zone(self, zone: int) -> np.memmap | None:
        if zone in self._cache:
            self._cache.move_to_end(zone)
            return self._cache[zone]

        path = self.zone_path(zone)
        if not path.exists():
            return None

        file_size = path.stat().st_size
        if file_size <= 0 or (file_size % _UCAC5_ZONE_RECORD_SIZE) != 0:
            raise ValueError(
                "UCAC5 zone file size is not a multiple of the record size "
                f"({_UCAC5_ZONE_RECORD_SIZE} bytes): {path} ({file_size} bytes)."
            )

        nrec = file_size // _UCAC5_ZONE_RECORD_SIZE
        if nrec <= 0:
            return None

        # Consistency check using ASC-derived counts (should match for correct catalogs)
        expected_nrec = int(self._zone_record_counts[zone - 1])
        if expected_nrec > 0 and nrec < expected_nrec:
            raise ValueError(
                "UCAC5 zone file appears truncated compared to u5index.asc: "
                f"{path} has {nrec} records, index expects at least {expected_nrec}."
            )

        mm = np.memmap(path, dtype=_UCAC5_U5Z_DTYPE, mode="r", shape=(nrec,))
        if mm.size == 0:
            return None

        self._cache[zone] = mm
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return mm


    @staticmethod
    def _wrap_ra_deg(ra_deg: float) -> float:
        ra = ra_deg % 360.0
        return ra + 360.0 if ra < 0.0 else ra

    @staticmethod
    def dec_to_zone(dec_deg: float) -> int:
        """
        Map Dec -> UCAC5 zone number 1..900.
        Zone 1: [-90.0, -89.8), zone 900: [89.8, 90.0)
        """
        d = max(-90.0, min(89.999999999, dec_deg))
        z0 = int(math.floor((d + 90.0) / _UCAC5_ZONE_HEIGHT_DEG))  # 0..899
        return z0 + 1

    @staticmethod
    def ra_to_bin(ra_deg: float) -> int:
        """
        Map RA -> UCAC5 bin number 1..1440 (0.25 deg per bin).
        Bin 1: [0.00, 0.25), bin 1440: [359.75, 360.00)
        """
        r = Ucac5Index._wrap_ra_deg(ra_deg)
        r = min(359.999999999, r)
        b0 = int(math.floor(r / _UCAC5_BIN_WIDTH_DEG))  # 0..1439
        return b0 + 1

    def get_start_count(self, zone: int, bin_: int) -> Tuple[int, int]:
        if not (1 <= zone <= _UCAC5_ZONES):
            raise ValueError(f"zone out of range: {zone}")
        if not (1 <= bin_ <= _UCAC5_BINS_PER_ZONE):
            raise ValueError(f"bin out of range: {bin_}")
        start, count = self._idx[zone - 1, bin_ - 1]
        return int(start), int(count)

    def spans_for_radec_box(
        self,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
    ) -> List[Ucac5Span]:
        """
        Return (zone,start,count) spans to read for an RA/Dec rectangle.
        Handles RA wraparound; merges adjacent spans per zone.
        """
        dec_lo = max(-90.0, min(dec_min_deg, dec_max_deg))
        dec_hi = min(90.0,  max(dec_min_deg, dec_max_deg))
        if dec_lo > dec_hi:
            return []

        z0 = self.dec_to_zone(dec_lo)
        z1 = self.dec_to_zone(min(89.999999999, dec_hi))

        # RA wrap handling: split into up to 2 intervals in [0,360)
        a0 = self._wrap_ra_deg(ra_min_deg)
        b0 = self._wrap_ra_deg(ra_max_deg)

        if a0 <= b0 and abs(ra_max_deg - ra_min_deg) < 360.0:
            intervals = [(a0, b0)]
        else:
            intervals = [(a0, 360.0), (0.0, b0)]

        spans: List[Ucac5Span] = []
        for zone in range(z0, z1 + 1):
            for a, b in intervals:
                a = max(0.0, min(359.999999999, a))
                b = max(0.0, min(359.999999999, b))

                bin_a = self.ra_to_bin(a)
                bin_b = self.ra_to_bin(b)

                # If the interval is tiny, still include that bin
                if a <= b:
                    for bin_ in range(bin_a, bin_b + 1):
                        start, count = self.get_start_count(zone, bin_)
                        if count > 0:
                            spans.append(Ucac5Span(zone=zone, start=start, count=count))

        # Merge adjacent spans within same zone to reduce reads
        spans.sort(key=lambda s: (s.zone, s.start))
        merged: List[Ucac5Span] = []
        for s in spans:
            if not merged or merged[-1].zone != s.zone:
                merged.append(s)
                continue
            prev = merged[-1]
            if prev.start + prev.count == s.start:
                merged[-1] = Ucac5Span(zone=prev.zone, start=prev.start, count=prev.count + s.count)
            else:
                merged.append(s)

        return merged

    def query_box(
        self,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
        *,
        thin: int = 1,
    ) -> Ucac5QueryResult:
        """Query catalog stars inside a RA/Dec rectangle in degrees."""

        thin = max(1, int(thin))
        dec_min_deg = max(-90.0, min(dec_min_deg, dec_max_deg))
        dec_max_deg = min(90.0, max(dec_min_deg, dec_max_deg))

        a0 = self._wrap_ra_deg(ra_min_deg)
        b0 = self._wrap_ra_deg(ra_max_deg)
        if a0 <= b0 and abs(ra_max_deg - ra_min_deg) < 360.0:
            ra_intervals = [(a0, b0)]
        else:
            ra_intervals = [(a0, 360.0), (0.0, b0)]

        spans = self.spans_for_radec_box(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg)
        if not spans:
            return Ucac5QueryResult(np.empty(0), np.empty(0))

        ra_out: list[np.ndarray] = []
        dec_out: list[np.ndarray] = []

        for span in spans:
            mm = self._load_zone(span.zone)
            if mm is None:
                continue
            start = span.start
            stop = start + span.count
            if stop <= start:
                continue
            records = mm[start:stop:thin]
            if records.size == 0:
                continue
            ra_deg = (records["ra_mas"].astype(np.float64) * _MAS_TO_DEG) % 360.0
            dec_deg = (records["dec_mas"].astype(np.float64) * _MAS_TO_DEG)
            mask = (dec_deg >= dec_min_deg) & (dec_deg <= dec_max_deg)
            if ra_intervals:
                ra_mask = np.zeros_like(mask, dtype=bool)
                for ra_start, ra_end in ra_intervals:
                    ra_mask |= (ra_deg >= ra_start) & (ra_deg <= ra_end)
                mask &= ra_mask
            if not np.any(mask):
                continue
            ra_out.append(ra_deg[mask])
            dec_out.append(dec_deg[mask])

        if not ra_out:
            return Ucac5QueryResult(np.empty(0), np.empty(0))

        return Ucac5QueryResult(np.concatenate(ra_out), np.concatenate(dec_out))
