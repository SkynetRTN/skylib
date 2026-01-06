from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

from skylib.astrometry.main import (
    AtlasConfig,
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


def _sample_image_path(sample: dict[str, object]) -> Path:
    image = sample.get("image")
    if not image:
        raise ValueError(f"Sample is missing the 'image' field")
    path = DATA_ROOT / str(image)
    if not path.exists():
        pytest.skip(f"Sample image missing: {path}")
    return path

def _sample_wcs_path(sample: dict[str, object]) -> Path:
    wcs = sample.get("wcs")
    if not wcs:
        raise ValueError(f"Sample is missing the 'wcs' field")
    path = DATA_ROOT / str(wcs)
    if not path.exists():
        pytest.skip(f"Sample WCS file missing: {path}")
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
    _maybe_save_sources(image_path, data, xy, label="an")
    height, width = data.shape[:2]
    return xy, flux, width, height


def _maybe_save_sources(image_path: Path, data: np.ndarray, xy: np.ndarray, *, label: str) -> None:
    if not os.getenv("SKYLIB_DEBUG_SOURCES"):
        return
    if importlib.util.find_spec("matplotlib") is None:
        return

    import matplotlib.pyplot as plt

    output_dir = Path(tempfile.gettempdir())
    output_path = output_dir / f"skylib_sources_{label}_{image_path.stem}.png"

    norm = ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch())
    plt.figure(figsize=(8, 8))
    plt.imshow(data, origin="lower", cmap="gray", norm=norm)
    plt.scatter(xy[:, 0] - 1, xy[:, 1] - 1, s=20, edgecolor="cyan", facecolor="none")
    plt.title(f"Extracted sources: {image_path.name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def test_solve_field_v2_astap_samples() -> None:
    cmd = os.getenv("SKYLIB_ASTAP_CMD", "astap_cli")
    if shutil.which(cmd) is None:
        pytest.skip(f"ASTAP executable not found: {cmd}")

    catalog = os.getenv("SKYLIB_ASTAP_CATALOG")
    if not catalog or not Path(catalog).exists():
        pytest.skip("ASTAP catalog path not configured or missing")

    samples = _get_samples("astap")
    config = AstapConfig(cmd=cmd, catalog=catalog)

    for sample in samples:
        image_path = _sample_image_path(sample)
        request = _solve_request_from_sample(sample, image_path)
        solution = solve_field_v2(request, backend="astap", configs={"astap": config})
        assert solution.backend == "astap"
        assert solution.wcs is not None, f"ASTAP failed to solve {image_path}"


def test_solve_field_v2_platesolve_samples(tmp_path: Path) -> None:
    cmd = os.getenv("SKYLIB_PLATESOLVE_CMD")
    if not cmd:
        pytest.skip("PlateSolve executable not configured")
    if shutil.which(cmd) is None:
        pytest.skip(f"PlateSolve executable not found: {cmd}")

    cwd = os.getenv("SKYLIB_PLATESOLVE_CWD")
    config = PlateSolveConfig(cmd=cmd, cwd=Path(cwd) if cwd else None)

    samples = _get_samples("platesolve")

    for sample in samples:
        image_path = _sample_image_path(sample)
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

    index_path = os.getenv("SKYLIB_ASTROMETRYNET_INDEX_PATH")
    if not index_path:
        pytest.skip("Astrometry.net index path not configured")

    index_paths = [path for path in index_path.split(os.pathsep) if path]
    if not all(Path(path).exists() for path in index_paths):
        pytest.skip("Astrometry.net index path does not exist")

    samples = _get_samples("an")
    config = AstrometryNetConfig(index_path=index_paths)

    for sample in samples:
        image_path = _sample_image_path(sample)
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


def test_solve_field_v2_atlas_samples() -> None:
    ucac4_root = os.getenv("SKYLIB_UCAC4_ROOT")
    if not ucac4_root or not Path(ucac4_root).exists():
        pytest.skip("UCAC4 root path not configured or missing")

    ucac5_root = os.getenv("SKYLIB_UCAC5_ROOT")
    if not ucac5_root or not Path(ucac5_root).exists():
        pytest.skip("UCAC5 root path not configured or missing")

    samples = _get_samples("atlas")
    config = AtlasConfig(catalog="ucac5",catalog_roots={"ucac4": Path(ucac4_root), "ucac5": Path(ucac5_root)}, debug=True)

    for sample in samples:
        image_path = _sample_image_path(sample)
        wcs_path = _sample_wcs_path(sample)
        request = _solve_request_from_sample(sample, image_path)
        solution = solve_field_v2(request, backend="atlas", configs={"atlas": config})
        assert solution.backend == "atlas"
        assert solution.wcs is not None, f"Atlas failed to solve {image_path}"
        
        print(f"Solved {image_path.name}: WCS = {solution.wcs}")
        assert_wcs_matches_reference(
            test_wcs=solution.wcs,
            reference_wcs_fits_path=wcs_path,
            width=1000,
            height=1000,
        )


def assert_wcs_matches_reference(
    test_wcs: WCS,
    reference_wcs_fits_path,
    *,
    width: int,
    height: int,
    n_grid: int = 7,                 # 7x7 grid across image
    max_sep_arcsec: float = 1.0,     # tolerance
    drop_nans: bool = True,
) -> None:
    """
    Assert that test_wcs and reference WCS from a FITS file agree to within
    max_sep_arcsec across a pixel grid spanning the image.
    """
    with fits.open(reference_wcs_fits_path) as hdul:
        ref_wcs = WCS(hdul[0].header)

    # Grid of pixel coordinates (0-based, since Astropy WCS uses 0-based in pixel_to_world_values)
    xs = np.linspace(0, width - 1, n_grid, dtype=float)
    ys = np.linspace(0, height - 1, n_grid, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    px = xx.ravel()
    py = yy.ravel()

    # Convert pixels -> world for both WCS
    ra1, dec1 = test_wcs.pixel_to_world_values(px, py)
    ra2, dec2 = ref_wcs.pixel_to_world_values(px, py)

    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame="icrs")
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame="icrs")

    sep = c1.separation(c2).to(u.arcsec).value

    if drop_nans:
        sep = sep[np.isfinite(sep)]

    assert sep.size > 0, "No finite comparison points for WCS match check"

    worst = float(np.nanmax(sep))
    assert worst <= max_sep_arcsec, (
        f"WCS mismatch: worst separation {worst:.3f} arcsec "
        f"(tolerance {max_sep_arcsec:.3f} arcsec)"
    )
        

if __name__ == "__main__":
    # test_solve_field_v2_astap_samples()
    # test_solve_field_v2_platesolve_samples(tmp_path=Path(tempfile.gettempdir()))
    # test_solve_field_v2_astrometry_net_samples()
    test_solve_field_v2_atlas_samples()