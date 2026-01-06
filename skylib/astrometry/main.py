"""SkyLib astrometric reduction package with backend support."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

from astropy.io import fits
from astropy.wcs import WCS

from .anet import (
    AstrometryNetBackend,
    AstrometryNetConfig,
    AstrometryNetSolver,
    an_engine,
    solve_field_glob as solve_field_glob_astrometry_net,
)
from .atlas import AtlasBackend, AtlasConfig
from .types import Backend, SolveRequest, SolveSolution

BackendConfig = Union[
    AstrometryNetConfig,
    AtlasConfig,
]


Solver = AstrometryNetSolver
Solution = SolveSolution


def solve_field(
    engine=None,
    xy=None,
    flux=None,
    width=None,
    height=None,
    ra_hours=0,
    dec_degs=0,
    radius=180,
    min_scale=0.1,
    max_scale=10,
    fov=None,
    parity=None,
    sip_order=3,
    crpix_center=True,
    max_sources=None,
    retry_lost=True,
    callback=None,
    backend=None,
    image_path: Path = None,
    downsample: Optional[int] = None,
    ucac4_root: Optional[Path] = None,
    ucac5_root: Optional[Path] = None,
    atlas_catalog: str = "ucac4",
    atlas_catalog_roots: Optional[Mapping[str, Path]] = None,
) -> SolveSolution:
    """Obtain astrometric solution given XY coordinates of field stars."""

    if backend is None:
        backend = "an"
    if backend not in {"an", "astrometry.net"}:
        raise ValueError(
            "solve_field only wraps the Astrometry.net backend; "
            "use AtlasBackend directly for Atlas solving",
        )
    if an_engine is None:
        raise ValueError("Astrometry.net backend is not available on this system")

    request = SolveRequest(
        xy=xy,
        flux=flux,
        width=width,
        height=height,
        ra_hours=ra_hours,
        dec_degs=dec_degs,
        radius=radius,
        min_scale=min_scale,
        max_scale=max_scale,
        fov=fov,
        parity=parity,
        sip_order=sip_order,
        crpix_center=crpix_center,
        max_sources=max_sources,
        retry_lost=retry_lost,
        callback=callback,
        image_path=image_path,
        downsample=downsample,
    )

    if engine is None:
        raise ValueError("engine must be provided for Astrometry.net backend")
    config = AstrometryNetConfig(engine=engine)
    backend_instance = AstrometryNetBackend()
    return backend_instance.solve(request, config)


def solve_field_glob(
    engine,
    xy,
    flux=None,
    width=None,
    height=None,
    ra_hours=0,
    dec_degs=0,
    radius=180,
    min_scale=0.1,
    max_scale=10,
    parity=None,
    sip_order=3,
    crpix_center=True,
    max_sources=None,
    retry_lost=True,
    callback=None,
    min_sources=10,
    initial_radius=1,
    radius_step=0.8,
) -> SolveSolution:
    """Obtain astrometric solution for fields around globular clusters."""

    request = SolveRequest(
        xy=xy,
        flux=flux,
        width=width,
        height=height,
        ra_hours=ra_hours,
        dec_degs=dec_degs,
        radius=radius,
        min_scale=min_scale,
        max_scale=max_scale,
        parity=parity,
        sip_order=sip_order,
        crpix_center=crpix_center,
        max_sources=max_sources,
        retry_lost=retry_lost,
        callback=callback,
    )

    config = AstrometryNetConfig(engine=engine)
    return solve_field_glob_astrometry_net(
        request,
        config,
        min_sources=min_sources,
        initial_radius=initial_radius,
        radius_step=radius_step,
    )


def _load_wcs(path: Path) -> Optional[WCS]:
    if not path.exists():
        return None
    try:
        header = fits.Header.fromtextfile(str(path))
        return WCS(header)
    except Exception:
        try:
            header = fits.getheader(str(path))
            return WCS(header)
        except Exception:
            return None


__all__ = [
    "AstrometryNetBackend",
    "AstrometryNetConfig",
    "AstrometryNetSolver",
    "Backend",
    "AtlasBackend",
    "AtlasConfig",
    "SolveRequest",
    "SolveSolution",
    "Solver",
    "Solution",
    "solve_field",
    "solve_field_glob",
    "an_engine",
]
