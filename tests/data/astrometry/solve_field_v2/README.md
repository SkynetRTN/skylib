# solve_field_v2 test data

Add sample FITS images under the backend-specific directories:

- `astap/`
- `platesolve/`
- `an/` (astrometry.net)

Update `samples.json` with the parameters needed to solve each image. Each entry
supports the following keys (all optional unless noted otherwise):

- `image` (required): filename within the backend directory.
- `ra_hours`, `dec_degs`: approximate field center.
- `radius`: search radius in degrees.
- `fov`: field of view in degrees.
- `min_scale`, `max_scale`: scale bounds used by the astrometry.net backend.
- `downsample`: integer downsample factor.
- `max_sources`: maximum extracted sources for the astrometry.net backend.
- `width`, `height`: explicit image dimensions when not read from the FITS data.

Example:

```json
{
  "astap": [
    {
      "image": "m31_astap.fits",
      "ra_hours": 0.712,
      "dec_degs": 41.269,
      "radius": 5.0,
      "fov": 2.0
    }
  ],
  "platesolve": [
    {
      "image": "m31_ps3.fits",
      "ra_hours": 0.712,
      "dec_degs": 41.269,
      "radius": 5.0,
      "fov": 2.0
    }
  ],
  "an": [
    {
      "image": "m31_an.fits",
      "ra_hours": 0.712,
      "dec_degs": 41.269,
      "radius": 5.0,
      "min_scale": 0.2,
      "max_scale": 2.0,
      "max_sources": 2000
    }
  ]
}
```

Environment variables used by the tests:

- `SKLIB_ASTAP_CMD`: path to `astap_cli` (defaults to `astap_cli`).
- `SKLIB_ASTAP_CATALOG`: path to ASTAP star catalog directory (required).
- `SKLIB_PLATESOLVE_CMD`: path to PlateSolve executable.
- `SKLIB_PLATESOLVE_CWD`: optional working directory for PlateSolve.
- `SKLIB_ASTROMETRYNET_INDEX_PATH`: one or more index paths (separated by your OS path separator).
