# solve_field_v2 test data

Add sample FITS images under the backend-specific directories:

- `an/` (astrometry.net)
- `atlas/`

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
  ],
  "atlas": [
    {
      "image": "m31_atlas.fits",
      "ra_hours": 0.712,
      "dec_degs": 41.269,
      "radius": 5.0,
      "fov": 2.0
    }
  ]
}
```

Environment variables used by the tests:

- `SKYLIB_ASTROMETRYNET_INDEX_PATH`: one or more index paths (separated by your OS path separator).
