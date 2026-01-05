"""SkyLib astrometric reduction package with backend support."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Sequence, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import Sip, WCS

from skylib.util.angle import angdist

from .atlas.solve.solver import solve as atlas_solve

try:  # pragma: no cover - optional dependency
    from . import an_engine
except Exception:  # pragma: no cover - missing optional dependency
    an_engine = None

BackendConfig = Union[
    "AstrometryNetConfig",
    "AstapConfig",
    "PlateSolveConfig",
    "AtlasConfig",
]


@dataclass(frozen=True)
class SolveRequest:
    xy: Optional[np.ndarray] = None
    flux: Optional[np.ndarray] = None
    width: Optional[int] = None
    height: Optional[int] = None
    ra_hours: float = 0.0
    dec_degs: float = 0.0
    radius: float = 180.0
    min_scale: float = 0.1
    max_scale: float = 10.0
    fov: Optional[float] = None
    parity: Optional[bool] = None
    sip_order: int = 3
    crpix_center: bool = True
    max_sources: Optional[int] = None
    retry_lost: bool = True
    callback: Optional[Callable[[], int]] = None
    image_path: Optional[Path] = None
    downsample: Optional[int] = None


@dataclass
class SolveSolution:
    wcs: Optional[WCS] = None
    log_odds: Optional[float] = None
    n_match: Optional[int] = None
    n_conflict: Optional[int] = None
    n_field: Optional[int] = None
    index_name: Optional[str] = None
    backend: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def solve(self, request: SolveRequest, config: Optional[BackendConfig]) -> SolveSolution: ...


@dataclass
class AstrometryNetConfig:
    index_path: Optional[Union[str, Sequence[str]]] = None
    engine: Optional["AstrometryNetSolver"] = None


@dataclass
class AstapConfig:
    cmd: str = "astap_cli"
    catalog: Optional[str] = "C:/astap"


@dataclass
class PlateSolveConfig:
    cmd: str = "platesolve3.80"
    args: Sequence[str] = (
        "{image_path}",
        "{ra_rad}",
        "{dec_rad}",
        "{x_size_rad}",
        "{y_size_rad}",
    )
    output_suffix: str = ".txt"
    output_path: Optional[Path] = None
    cwd: Optional[Path] = None


@dataclass
class AtlasConfig:
    ucac4_root: Optional[Path] = None
    ucac5_root: Optional[Path] = None
    catalog: str = "ucac4"
    catalog_roots: Mapping[str, Path] = field(default_factory=dict)
    timeout_s: Optional[float] = None
    max_catalog_stars: int = 400
    max_image_stars: int = 120
    n_tri_obs: int = 8000
    n_tri_cat: int = 15000
    invariant_tol: float = 0.006
    match_tol_arcsec: float = 6.0
    refine_center: bool = True
    thin: int = 1

    def resolve_catalog(self) -> tuple[str, Path]:
        catalog = self.catalog.strip().lower()
        if catalog in self.catalog_roots:
            return catalog, self.catalog_roots[catalog]
        if catalog == "ucac4":
            if self.ucac4_root is None:
                raise ValueError("ucac4_root must be provided for UCAC4 catalog")
            return catalog, self.ucac4_root
        if catalog == "ucac5":
            if self.ucac5_root is None:
                raise ValueError("ucac5_root must be provided for UCAC5 catalog")
            return catalog, self.ucac5_root
        raise ValueError(f"Unsupported catalog: {self.catalog}")


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


class AstrometryNetBackend:
    name = "an"

    def is_available(self) -> bool:
        return an_engine is not None

    def solve(self, request: SolveRequest, config: Optional[BackendConfig]) -> SolveSolution:
        if request.xy is None:
            raise ValueError("xy must be provided")

        if not isinstance(config, AstrometryNetConfig):
            raise ValueError("Astrometry.net config is required")

        engine = config.engine
        if engine is None:
            if config.index_path is None:
                raise ValueError("index_path or engine must be provided")
            engine = AstrometryNetSolver(config.index_path)

        solver = engine.solver
        ra = float(request.ra_hours) * 15
        dec = float(request.dec_degs)
        r = float(request.radius)

        if request.callback is not None:
            if sys.platform.startswith("win32"):
                time_t = ctypes.c_int64
            elif ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_int64):
                time_t = ctypes.c_int64
            else:
                time_t = ctypes.c_int32
            an_engine.set_timer_callback(
                solver,
                ctypes.cast(ctypes.CFUNCTYPE(time_t)(request.callback), ctypes.c_voidp).value,
            )
        else:
            an_engine.set_timer_callback(solver, 0)

        n = len(request.xy)
        xy = np.asanyarray(request.xy)
        field = an_engine.starxy_new(n, request.flux is not None, False)
        if request.flux is not None:
            flux = np.asanyarray(request.flux)
            if len(flux) != n:
                raise ValueError("Flux array must be of the same length as XY array")
            if request.max_sources:
                order = np.argsort(flux)[::-1]
                xy, flux = xy[order], flux[order]
                del order
            an_engine.starxy_set_flux_array(field, flux)
        an_engine.starxy_set_xy_array(field, xy.ravel())
        an_engine.solver_set_field(solver, field)

        try:
            if request.width:
                minx, maxx = 1, int(request.width)
            else:
                minx, maxx = xy[:, 0].min(), xy[:, 0].max()
            if request.height:
                miny, maxy = 1, int(request.height)
            else:
                miny, maxy = xy[:, 1].min(), xy[:, 1].max()
            an_engine.solver_set_field_bounds(solver, minx, maxx, miny, maxy)
            solver.quadsize_min = 0.1 * min(maxx - minx + 1, maxy - miny + 1)

            if request.crpix_center != "":
                solver.set_crpix = solver.set_crpix_center = int(request.crpix_center)

            an_engine.solver_set_radec(solver, ra, dec, r)

            solver.funits_lower = float(request.min_scale)
            solver.funits_upper = float(request.max_scale)

            solver.logratio_tokeep = np.log(1e12)
            solver.distance_from_quad_bonus = True

            if request.parity is None or request.parity == "":
                solver.parity = an_engine.PARITY_BOTH
            elif int(request.parity):
                solver.parity = an_engine.PARITY_NORMAL
            else:
                solver.parity = an_engine.PARITY_FLIP

            enable_sip = request.sip_order and int(request.sip_order) >= 2
            if enable_sip:
                solver.do_tweak = True
                solver.tweak_aborder = int(request.sip_order)
                solver.tweak_abporder = int(request.sip_order) + 1
            else:
                solver.do_tweak = False

            if request.max_sources:
                solver.endobj = request.max_sources
            else:
                solver.endobj = 0

            if request.width is None or request.height is None:
                width = maxx - minx + 1
                height = maxy - miny + 1
            else:
                width = request.width
                height = request.height

            fmin = solver.quadsize_min * request.min_scale
            fmax = np.hypot(width, height) * request.max_scale
            indices = []
            for index in engine.indexes:
                if fmin > index.index_scale_upper or fmax < index.index_scale_lower:
                    continue
                if not an_engine.index_is_within_range(index, ra, dec, r):
                    continue

                indices.append(index)

            if not indices:
                raise ValueError("No indexes found for the given scale and position")

            indices.sort(
                key=lambda _idx: (
                    -_idx.index_scale_upper,
                    an_engine.healpix_distance_to_radec(
                        _idx.healpix,
                        _idx.hpnside,
                        ra,
                        dec,
                    )[0]
                    if _idx.healpix >= 0
                    else 0,
                )
            )
            an_engine.solver_clear_indexes(solver)
            for index in indices:
                an_engine.solver_add_index(solver, index)

            an_engine.solver_run(solver)
            sol = SolveSolution(backend=self.name)

            if solver.have_best_match:
                best_match = solver.best_match
                sol.log_odds = solver.best_logodds
                sol.n_match = best_match.nmatch
                sol.n_conflict = best_match.nconflict
                sol.n_field = best_match.nfield
                if best_match.index is not None:
                    sol.index_name = best_match.index.indexname
            else:
                best_match = None

            if solver.best_match_solves:
                sol.wcs = WCS(naxis=2)

                wcs_ctype = ("RA---TAN", "DEC--TAN")
                if enable_sip:
                    sip = best_match.sip
                    try:
                        wcstan = sip.wcstan
                    except AttributeError:
                        wcstan = best_match.wcstan
                    else:
                        a_order, b_order = sip.a_order, sip.b_order
                        if a_order > 0 or b_order > 0:
                            ap_order, bp_order = sip.ap_order, sip.bp_order
                            maxorder = an_engine.SIP_MAXORDER
                            a = array_from_swig(sip.a, (maxorder, maxorder))[: a_order + 1, : a_order + 1]
                            b = array_from_swig(sip.b, (maxorder, maxorder))[: b_order + 1, : b_order + 1]
                            if a.any() or b.any():
                                ap = array_from_swig(sip.ap, (maxorder, maxorder))[
                                    : ap_order + 1,
                                    : ap_order + 1,
                                ]
                                bp = array_from_swig(sip.bp, (maxorder, maxorder))[
                                    : bp_order + 1,
                                    : bp_order + 1,
                                ]
                                sol.wcs.sip = Sip(
                                    a,
                                    b,
                                    ap,
                                    bp,
                                    array_from_swig(wcstan.crpix, (2,)),
                                )
                                wcs_ctype = ("RA---TAN-SIP", "DEC--TAN-SIP")
                else:
                    wcstan = best_match.wcstan
                sol.wcs.wcs.ctype = wcs_ctype
                sol.wcs.wcs.crpix = array_from_swig(wcstan.crpix, (2,))
                sol.wcs.wcs.crval = array_from_swig(wcstan.crval, (2,))
                sol.wcs.wcs.cd = array_from_swig(wcstan.cd, (2, 2))
            elif request.retry_lost and (request.radius < 180 or request.parity is not None):
                an_engine.solver_cleanup_field(solver)
                retry_request = SolveRequest(
                    xy=request.xy,
                    flux=request.flux,
                    width=request.width,
                    height=request.height,
                    ra_hours=0,
                    dec_degs=0,
                    radius=180,
                    min_scale=request.min_scale,
                    max_scale=request.max_scale,
                    fov=None,
                    parity=None,
                    sip_order=request.sip_order,
                    crpix_center=request.crpix_center,
                    max_sources=request.max_sources,
                    retry_lost=False,
                    callback=request.callback,
                )
                return self.solve(retry_request, config)

            return sol
        finally:
            an_engine.solver_cleanup_field(solver)
            an_engine.solver_clear_indexes(solver)


class AstapBackend:
    name = "astap"

    def is_available(self) -> bool:
        return True

    def solve(self, request: SolveRequest, config: Optional[BackendConfig]) -> SolveSolution:
        if not isinstance(config, AstapConfig):
            raise ValueError("ASTAP config is required")
        if request.image_path is None:
            raise ValueError("image_path must be provided for ASTAP backend")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "solved"

            cmdline = [config.cmd, "-f", str(request.image_path), "-o", str(output)]
            if request.ra_hours is not None:
                cmdline.extend(["-ra", str(float(request.ra_hours))])
            if request.dec_degs is not None:
                cmdline.extend(["-spd", str(float(request.dec_degs + 90.0))])
            if request.radius is not None:
                cmdline.extend(["-r", str(float(request.radius))])
            if request.fov is not None:
                cmdline.extend(["-fov", str(float(request.fov))])
            if config.catalog:
                cmdline.extend(["-d", config.catalog])
            if request.downsample is not None:
                cmdline.extend(["-z", str(int(request.downsample))])

            subprocess.run(cmdline, check=False)

            sol = SolveSolution(backend=self.name)
            wcs_path = output.with_suffix(".wcs")
            sol.wcs = _load_wcs(wcs_path)
            return sol


class PlateSolveBackend:
    name = "platesolve"

    def is_available(self) -> bool:
        return True

    def solve(self, request: SolveRequest, config: Optional[BackendConfig]) -> SolveSolution:
        if not isinstance(config, PlateSolveConfig):
            raise ValueError("PlateSolve config is required")
        if request.image_path is None:
            raise ValueError("image_path must be provided for PlateSolve backend")

        output_path = config.output_path
        if output_path is None:
            output_path = request.image_path.with_stem(request.image_path.stem + "_PS3") \
                                            .with_suffix(config.output_suffix)

        ra_rad = np.deg2rad(float(request.ra_hours) * 15.0)
        dec_rad = np.deg2rad(float(request.dec_degs))
        if request.fov is not None:
            x_size = float(request.fov)
        elif request.radius is not None:
            x_size = float(request.radius) * 2.0
        else:
            x_size = 0.0
        y_size = x_size
        if request.width and request.height and x_size:
            y_size = x_size * (float(request.height) / float(request.width))
        x_size_rad = np.deg2rad(x_size)
        y_size_rad = np.deg2rad(y_size)

        with tempfile.TemporaryDirectory() as tmp:
            temp_image_path = _inject_platesolve_wcs(
                request.image_path,
                Path(tmp),
                request,
                x_size=x_size,
                y_size=y_size,
            )

            context = {
                "cmd": config.cmd,
                "image_path": temp_image_path,
                "ra_hours": request.ra_hours,
                "dec_degs": request.dec_degs,
                "radius": request.radius,
                "fov": request.fov,
                "width": request.width,
                "height": request.height,
                "output_path": output_path,
                "downsample": request.downsample,
                "ra_rad": ra_rad,
                "dec_rad": dec_rad,
                "x_size_rad": x_size_rad,
                "y_size_rad": y_size_rad,
            }
            cmdline = [config.cmd]
            for arg in config.args:
                formatted = arg.format_map(context)
                if formatted:
                    cmdline.append(formatted)

            subprocess.run(cmdline, check=False, cwd=str(config.cwd) if config.cwd else None)

        sol = SolveSolution(backend=self.name)
        if output_path.suffix.lower() == ".txt":
            sol.wcs, sol.metadata = _load_platesolve_solution(output_path)
        else:
            sol.wcs = _load_wcs(output_path)
        return sol


class AtlasBackend:
    name = "atlas"

    def is_available(self) -> bool:
        return True

    def solve(self, request: SolveRequest, config: Optional[BackendConfig]) -> SolveSolution:
        if not isinstance(config, AtlasConfig):
            raise ValueError("Atlas config is required")
        if request.image_path is None:
            raise ValueError("image_path must be provided for Atlas backend")

        catalog, catalog_root = config.resolve_catalog()


        fov_guess = None
        if request.fov is not None:
            fov_guess = (float(request.fov), float(request.fov))

        result = atlas_solve(
            request.image_path,
            catalog_root,
            catalog=catalog,
            ra0_deg=float(request.ra_hours) * 15.0,
            dec0_deg=float(request.dec_degs),
            scale_range_arcsec_per_pix=(float(request.min_scale), float(request.max_scale)),
            fov_guess_deg=fov_guess,
            timeout_s=config.timeout_s,
            max_catalog_stars=config.max_catalog_stars,
            max_image_stars=config.max_image_stars,
            n_tri_obs=config.n_tri_obs,
            n_tri_cat=config.n_tri_cat,
            invariant_tol=config.invariant_tol,
            match_tol_arcsec=config.match_tol_arcsec,
            refine_center=config.refine_center,
            thin=config.thin,
        )

        sol = SolveSolution(backend=self.name)
        sol.wcs = result.wcs
        sol.metadata = result.metadata
        return sol


def _inject_platesolve_wcs(
    image_path: Path,
    tmp_dir: Path,
    request: SolveRequest,
    *,
    x_size: float,
    y_size: float,
) -> Path:
    if x_size <= 0 or y_size <= 0:
        return image_path

    try:
        with fits.open(image_path) as hdul:
            hdu = hdul[0]
            data = hdu.data
            header = hdu.header.copy()
    except Exception:
        return image_path

    if data is None or data.ndim < 2:
        return image_path

    width = request.width or data.shape[1]
    height = request.height or data.shape[0]
    if width <= 0 or height <= 0:
        return image_path

    ra_deg = float(request.ra_hours) * 15.0
    dec_deg = float(request.dec_degs)
    scale_x = x_size / float(width)
    scale_y = y_size / float(height)
    crpix1 = (float(width) + 1.0) / 2.0
    crpix2 = (float(height) + 1.0) / 2.0

    header["WCSAXES"] = 2
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = ra_deg
    header["CRVAL2"] = dec_deg
    header["CRPIX1"] = crpix1
    header["CRPIX2"] = crpix2
    header["CDELT1"] = -scale_x
    header["CDELT2"] = scale_y
    header["EQUINOX"] = 2000.0

    output_path = tmp_dir / f"{image_path.stem}_platesolve_wcs.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(output_path, overwrite=True)
    return output_path


def solve_field_v2(
    request: SolveRequest,
    backend: Optional[Union[str, Backend]] = None,
    preferred_backends: Optional[Sequence[Union[str, Backend]]] = None,
    configs: Optional[Mapping[str, BackendConfig]] = None,
) -> SolveSolution:
    registry = {
        "an": AstrometryNetBackend(),
        "astrometry.net": AstrometryNetBackend(),
        "astap": AstapBackend(),
        "platesolve": PlateSolveBackend(),
        "atlas": AtlasBackend(),
    }

    if configs is None:
        configs = {}

    def resolve_backend(item: Union[str, Backend]) -> Backend:
        if isinstance(item, str):
            if item not in registry:
                raise ValueError(f"Unknown backend: {item}")
            return registry[item]
        return item

    if backend is not None:
        candidate = resolve_backend(backend)
        if not candidate.is_available():
            raise ValueError(f"Requested backend is not available: {candidate.name}")
        selected = [candidate]
    elif preferred_backends is not None:
        selected = [resolve_backend(item) for item in preferred_backends]
    else:
        default = ["an", "astap", "platesolve"] if an_engine is not None else ["astap", "platesolve"]
        selected = [resolve_backend(item) for item in default]

    last_solution = SolveSolution()
    for candidate in selected:
        if not candidate.is_available():
            continue
        cfg = configs.get(candidate.name)
        if cfg is None and candidate.name == "an":
            cfg = configs.get("astrometry.net")
        last_solution = candidate.solve(request, cfg)
        if last_solution.wcs is not None:
            return last_solution

    return last_solution


def solve_field_glob_v2(
    request: SolveRequest,
    config: AstrometryNetConfig,
    min_sources: int = 10,
    initial_radius: float = 1,
    radius_step: float = 0.8,
) -> SolveSolution:
    backend = AstrometryNetBackend()
    sol = backend.solve(request, config)
    engine = config.engine
    if engine is None:
        if config.index_path is None:
            raise ValueError("index_path or engine must be provided")
        engine = AstrometryNetSolver(config.index_path)

    if sol.wcs is not None and request.xy is not None and len(request.xy) >= min_sources:
        n = len(request.xy)
        xy = np.asarray(request.xy)
        flux = np.asarray(request.flux) if request.flux is not None else np.zeros(n)
        ra, dec = sol.wcs.all_pix2world(xy[:, 0], xy[:, 1], 1)
        ra %= 360
        ra /= 15
        radius = (dec.max() - dec.min()) / 2
        for ra0, dec0, r0 in engine.globs:
            r = r0 * initial_radius
            found = False
            prev_num_outer = None
            while True:
                inner = angdist(ra0, dec0, ra, dec) < r
                num_inner = inner.sum()
                if not num_inner:
                    break

                found = True

                outer = ~inner
                num_outer = n - num_inner
                if num_outer >= min_sources and num_outer != prev_num_outer:
                    prev_num_outer = num_outer
                    new_request = SolveRequest(
                        xy=xy[outer],
                        flux=flux[outer],
                        width=request.width,
                        height=request.height,
                        ra_hours=sol.wcs.wcs.crval[0] / 15,
                        dec_degs=sol.wcs.wcs.crval[1],
                        radius=radius,
                        min_scale=request.min_scale,
                        max_scale=request.max_scale,
                        parity=request.parity,
                        sip_order=0,
                        crpix_center=request.crpix_center,
                        max_sources=request.max_sources,
                        retry_lost=False,
                        callback=request.callback,
                    )
                    new_sol = backend.solve(new_request, config)
                    if new_sol.wcs is not None:
                        sol = new_sol
                        break

                r *= radius_step
            if found:
                break
    return sol


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
    astap_cmd="astap_cli",
    astap_catalog="C:/astap",
    image_path: Path = None,
    downsample: Optional[int] = None,
    ucac4_root: Optional[Path] = None,
    ucac5_root: Optional[Path] = None,
    atlas_catalog: str = "ucac4",
    atlas_catalog_roots: Optional[Mapping[str, Path]] = None,
) -> SolveSolution:
    """Obtain astrometric solution given XY coordinates of field stars."""

    if backend is None:
        backend = "an" if an_engine is not None else "astap"
    elif backend in {"an", "astrometry.net"} and an_engine is None:
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

    configs = {}
    if backend in {"an", "astrometry.net"}:
        if engine is None:
            raise ValueError("engine must be provided for Astrometry.net backend")
        configs["an"] = AstrometryNetConfig(engine=engine)
    elif backend == "astap":
        configs["astap"] = AstapConfig(cmd=astap_cmd, catalog=astap_catalog)
    elif backend == "atlas":
        configs["atlas"] = AtlasConfig(
            ucac4_root=ucac4_root,
            ucac5_root=ucac5_root,
            catalog=atlas_catalog,
            catalog_roots=atlas_catalog_roots or {},
        )

    return solve_field_v2(request, backend=backend, configs=configs)


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
    return solve_field_glob_v2(
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


def _load_platesolve_solution(path: Path) -> tuple[Optional[WCS], dict]:
    if not path.exists():
        return None, {}

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return None, {}

    metadata: dict = {}
    success_tokens = lines[0].lower().replace(",", " ").split()
    success = any(token == "true" for token in success_tokens)
    metadata["success"] = success
    if not success:
        return None, metadata

    def parse_floats(line: str) -> list[float]:
        cleaned = line.replace(",", " ")
        values = []
        for token in cleaned.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    ra_dec = parse_floats(lines[1]) if len(lines) > 1 else []
    if len(ra_dec) >= 2:
        ra_rad, dec_rad = ra_dec[0], ra_dec[1]
    else:
        return None, metadata
    metadata["ra_rad"] = ra_rad
    metadata["dec_rad"] = dec_rad

    scale_rot = parse_floats(lines[2]) if len(lines) > 2 else []
    if len(scale_rot) >= 2:
        imscale, rot_deg = scale_rot[0], scale_rot[1]
        metadata["imscale"] = imscale
        metadata["rotation_deg"] = rot_deg
    else:
        return None, metadata

    if len(lines) > 3:
        metadata["match_method"] = lines[3]

    if len(lines) > 4:
        coeffs = parse_floats(lines[4])
        if len(coeffs) >= 8:
            metadata["transform_coeffs"] = coeffs[:8]

    if len(lines) > 5:
        u0v0 = parse_floats(lines[5])
        if len(u0v0) >= 2:
            u0, v0 = u0v0[0], u0v0[1]
        else:
            u0, v0 = 0.0, 0.0
    else:
        u0, v0 = 0.0, 0.0
    metadata["u0"] = u0
    metadata["v0"] = v0

    if len(lines) > 6:
        extra = parse_floats(lines[6])
        if extra:
            metadata["extra"] = extra

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")
    wcs.wcs.crval = [np.rad2deg(ra_rad), np.rad2deg(dec_rad)]
    wcs.wcs.crpix = [u0, v0]
    scale_deg = (1.0 / imscale) * (180.0 / np.pi)
    theta = np.deg2rad(rot_deg)
    wcs.wcs.cd = scale_deg * np.array(
        [
            [-np.cos(theta), np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return wcs, metadata


def array_from_swig(data, shape, dtype=np.float64):
    a = np.empty(shape, dtype)
    ctypes.memmove(a.ctypes, int(data), a.nbytes)
    return a


__all__ = [
    "AstapBackend",
    "AstapConfig",
    "AstrometryNetBackend",
    "AstrometryNetConfig",
    "AstrometryNetSolver",
    "Backend",
    "AtlasBackend",
    "PlateSolveBackend",
    "PlateSolveConfig",
    "AtlasConfig",
    "SolveRequest",
    "SolveSolution",
    "Solver",
    "Solution",
    "solve_field",
    "solve_field_glob",
    "solve_field_v2",
    "solve_field_glob_v2",
    "an_engine",
]
