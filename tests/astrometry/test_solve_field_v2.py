from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from skylib.astrometry.main import (
    AstapConfig,
    AstrometryNetBackend,
    AstrometryNetConfig,
    PlateSolveConfig,
    SolveRequest,
    solve_field_v2,
)
from skylib.extraction.main import extract_sources

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "astrometry" / "solve_field_v2"
SAMPLES_PATH = DATA_ROOT / "samples.json"


def _load_samples() -> dict[str, list[dict[str, object]]]:
    if not SAMPLES_PATH.exists():
        pytest.skip(f"Missing sample metadata file: {SAMPLES_PATH}")
    with SAMPLES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_samples(backend: str) -> list[dict[str, object]]:
    data = _load_samples()
    samples = data.get(backend, [])
    if not samples:
        pytest.skip(f"No samples configured for backend '{backend}'")
    return samples


def _sample_image_path(backend: str, sample: dict[str, object]) -> Path:
    image = sample.get("image")
    if not image:
        raise ValueError(f"Sample for {backend} is missing the 'image' field")
    path = DATA_ROOT / backend / str(image)
    if not path.exists():
        pytest.skip(f"Sample image missing: {path}")
    return path


def _solve_request_from_sample(sample: dict[str, object], image_path: Path) -> SolveRequest:
    return SolveRequest(
        image_path=image_path,
        ra_hours=float(sample.get("ra_hours", 0.0)),
        dec_degs=float(sample.get("dec_degs", 0.0)),
        radius=float(sample.get("radius", 180.0)),
        fov=sample.get("fov"),
        min_scale=float(sample.get("min_scale", 0.1)),
        max_scale=float(sample.get("max_scale", 10.0)),
        downsample=sample.get("downsample"),
        width=sample.get("width"),
        height=sample.get("height"),
        max_sources=sample.get("max_sources"),
    )


def _extract_xy(image_path: Path, max_sources: int | None) -> tuple[np.ndarray, np.ndarray, int, int]:
    with fits.open(image_path) as hdul:
        data = hdul[0].data

    if data is None or data.ndim < 2:
        pytest.skip(f"Invalid image data for source extraction: {image_path}")

    sources, _, _ = extract_sources(data, max_sources=max_sources or 5000)
    if len(sources) == 0:
        pytest.skip(f"No sources extracted from {image_path}")

    xy = np.column_stack((sources["x"], sources["y"]))
    flux = sources["flux"]
    height, width = data.shape[:2]
    return xy, flux, width, height


def test_solve_field_v2_astap_samples() -> None:
    cmd = os.getenv("SKLIB_ASTAP_CMD", "astap_cli")
    if shutil.which(cmd) is None:
        pytest.skip(f"ASTAP executable not found: {cmd}")

    catalog = os.getenv("SKLIB_ASTAP_CATALOG")
    if not catalog or not Path(catalog).exists():
        pytest.skip("ASTAP catalog path not configured or missing")

    samples = _get_samples("astap")
    config = AstapConfig(cmd=cmd, catalog=catalog)

    for sample in samples:
        image_path = _sample_image_path("astap", sample)
        request = _solve_request_from_sample(sample, image_path)
        solution = solve_field_v2(request, backend="astap", configs={"astap": config})
        assert solution.backend == "astap"
        assert solution.wcs is not None, f"ASTAP failed to solve {image_path}"


def test_solve_field_v2_platesolve_samples(tmp_path: Path) -> None:
    cmd = os.getenv("SKLIB_PLATESOLVE_CMD")
    if not cmd:
        pytest.skip("PlateSolve executable not configured")
    if shutil.which(cmd) is None:
        pytest.skip(f"PlateSolve executable not found: {cmd}")

    cwd = os.getenv("SKLIB_PLATESOLVE_CWD")
    config = PlateSolveConfig(cmd=cmd, cwd=Path(cwd) if cwd else None)

    samples = _get_samples("platesolve")

    for sample in samples:
        image_path = _sample_image_path("platesolve", sample)
        temp_image = tmp_path / image_path.name
        temp_image.write_bytes(image_path.read_bytes())

        request = _solve_request_from_sample(sample, temp_image)
        solution = solve_field_v2(request, backend="platesolve", configs={"platesolve": config})
        assert solution.backend == "platesolve"
        assert solution.wcs is not None, f"PlateSolve failed to solve {image_path}"


def test_solve_field_v2_astrometry_net_samples() -> None:
    backend = AstrometryNetBackend()
    if not backend.is_available():
        pytest.skip("Astrometry.net engine is not available")

    index_path = os.getenv("SKLIB_ASTROMETRYNET_INDEX_PATH")
    if not index_path:
        pytest.skip("Astrometry.net index path not configured")

    index_paths = [path for path in index_path.split(os.pathsep) if path]
    if not all(Path(path).exists() for path in index_paths):
        pytest.skip("Astrometry.net index path does not exist")

    samples = _get_samples("an")
    config = AstrometryNetConfig(index_path=index_paths)

    for sample in samples:
        image_path = _sample_image_path("an", sample)
        request = _solve_request_from_sample(sample, image_path)
        xy, flux, width, height = _extract_xy(image_path, request.max_sources)
        request = SolveRequest(
            xy=xy,
            flux=flux,
            width=request.width or width,
            height=request.height or height,
            ra_hours=request.ra_hours,
            dec_degs=request.dec_degs,
            radius=request.radius,
            min_scale=request.min_scale,
            max_scale=request.max_scale,
            parity=request.parity,
            sip_order=request.sip_order,
            crpix_center=request.crpix_center,
            max_sources=request.max_sources,
            retry_lost=request.retry_lost,
            callback=request.callback,
        )
        solution = solve_field_v2(request, backend="an", configs={"an": config})
        assert solution.backend == "an"
        assert solution.wcs is not None, f"Astrometry.net failed to solve {image_path}"
