# skylib
A Python library for analyzing and processing astronomical image data

## ASTAP backend

The astrometric routines can optionally use the [ASTAP](https://www.hnsky.org/astap.htm)
solver. The ASTAP executable and its star catalog are **not** bundled with
SkyLib and must be installed separately. When using this backend supply the
paths to the executable and catalog via ``solve_field(..., backend="astap",
astap_cmd=..., astap_catalog=...)``.
