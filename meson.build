project(
    'skylib',
    'c',
    version: '2.0.0',
    meson_version: '>=1.6.0',
    default_options: [
        'buildtype=release',
        'c_std=c99',
        'warning_level=3',
    ]
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

py_sources = files(
    'skylib/__init__.py',
    'skylib/astrometry/__init__.py',
    'skylib/astrometry/an_engine.py',
    'skylib/astrometry/main.py',
    'skylib/calibration/__init__.py',
    'skylib/calibration/background.py',
    'skylib/calibration/bias.py',
    'skylib/calibration/cosmetic.py',
    'skylib/calibration/cosmic.py',
    'skylib/calibration/dark.py',
    'skylib/calibration/flat.py',
    'skylib/color/__init__.py',
    'skylib/color/radio.py',
    'skylib/combine/__init__.py',
    'skylib/combine/alignment.py',
    'skylib/combine/cropping.py',
    'skylib/combine/mosaicing.py',
    'skylib/combine/pattern_matching.py',
    'skylib/combine/smart_stacking.py',
    'skylib/combine/stacking.py',
    'skylib/combine/util.py',
    'skylib/enhancement/__init__.py',
    'skylib/enhancement/wavelet.py',
    'skylib/extraction/__init__.py',
    'skylib/extraction/centroiding.py',
    'skylib/extraction/main.py',
    'skylib/io/__init__.py',
    'skylib/io/conversion.py',
    'skylib/photometry/__init__.py',
    'skylib/photometry/aperture.py',
    'skylib/photometry/aperture_numba.py',
    'skylib/sonification/__init__.py',
    'skylib/sonification/main.py',
    'skylib/util/__init__.py',
    'skylib/util/angle.py',
    'skylib/util/fits.py',
    'skylib/util/overlap.py',
    'skylib/util/stats.py',
)
py.install_sources(py_sources)
install_data(
    'skylib/astrometry/ngc2000.dat',
    'skylib/sonification/start.wav',
    'skylib/sonification/stop.wav',
)
install_data('scripts/fits2wav.py', install_dir: 'bin')

add_project_arguments('-DNPY_NO_DEPRECATED_API=0', language : 'c')
add_project_arguments('-DNPY_TARGET_VERSION=NPY_1_26_API_VERSION', language : 'c')

windows = host_machine.system() == 'windows'

tparty = 'skylib/thirdparty/'
anet = tparty + 'astrometry.net/'
extra = tparty + 'anet_extra/'
gsl = anet + 'gsl-an/'

# Astrometry.net source files
an_engine_sources = ['skylib/astrometry/an_engine_wrap.c']
foreach fn: [
        'an-endian', 'bl', 'codekd', 'datalog', 'errors', 'fit-wcs', 'fitsbin', 'fitsfile', 'fitsioutils', 'fitstable',
        'gslutils', 'healpix', 'index', 'ioutils', 'log', 'matchobj', 'mathutil', 'permutedsort', 'quadfile', 'sip',
        'sip-utils', 'starkd', 'starutil', 'starxy', 'tic',
        ]
    an_engine_sources += anet + 'util/' + fn + '.c'
endforeach

foreach fn: ['quad-utils', 'solver', 'tweak', 'tweak2', 'verify']
    an_engine_sources += anet + 'solver/' + fn + '.c'
endforeach

# Libraries
gsl_an_sources = [gsl + '/blas/blas.c']
foreach fn: ['bidiag', 'cholesky', 'householder', 'lu', 'qr', 'svd']
    gsl_an_sources += gsl + 'linalg/' + fn + '.c'
endforeach
foreach fn: ['broyden', 'convergence', 'dnewton', 'fdfsolver', 'fdjac', 'fsolver', 'gnewton', 'hybrid', 'hybridj', 'newton']
    gsl_an_sources += gsl + 'multiroots/' + fn + '.c'
endforeach
foreach fn: ['block', 'init']
    gsl_an_sources += gsl + 'block/' + fn + '.c'
endforeach
foreach fn: ['copy', 'init', 'matrix', 'rowcol', 'submatrix', 'swap', 'view']
    gsl_an_sources += gsl + 'matrix/' + fn + '.c'
endforeach
foreach fn: ['copy', 'init', 'oper', 'prop', 'subvector', 'swap', 'vector']
    gsl_an_sources += gsl + 'vector/' + fn + '.c'
endforeach
foreach fn: ['init', 'permutation', 'permute']
    gsl_an_sources += gsl + 'permutation/' + fn + '.c'
endforeach
foreach subdir: ['sys', 'cblas', 'err']
    gsl_an_sources += run_command(
        py,
        ['-c', 'from glob import glob; print("\\n".join(glob("' + gsl + subdir + '/*.c")), end="")'],
        check: true).stdout().split('\n')
endforeach

kd_sources = []
foreach fn: ['dualtree', 'dualtree_nearestneighbour', 'dualtree_rangesearch', 'kdint_ddd', 'kdint_dds', 'kdint_ddu',
             'kdint_dss', 'kdint_duu', 'kdint_fff', 'kdint_lll', 'kdtree', 'kdtree_dim', 'kdtree_fits_io', 'kdtree_mem']
    kd_sources += anet + 'libkd/' + fn + '.c'
endforeach
kd_include_dirs = [extra + 'anet']
foreach fn: ['include', 'include/astrometry', 'qfits-an', 'util']
    kd_include_dirs += anet + fn
endforeach

qfits_an_sources = run_command(
    py,
    ['-c', 'from glob import glob; print("\\n".join(glob("' + anet + 'qfits-an/*.c")), end="")'],
    check: true).stdout().split('\n')
qfits_an_include_dirs = []
foreach fn: ['include', 'include/astrometry', 'qfits-an', 'util']
    qfits_an_include_dirs += anet + fn
endforeach

libs = [
    static_library(
        'gsl-an',
        gsl_an_sources,
        include_directories: include_directories([gsl, extra + 'gsl-an']),
    ),
    static_library(
        'kd',
        kd_sources,
        include_directories: include_directories(kd_include_dirs),
    ),
    static_library(
        'qfits-an',
        qfits_an_sources,
        include_directories: include_directories(qfits_an_include_dirs),
    ),
]

# Global include directories
np_incdir = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
include_dirs = [np_incdir]
foreach f: ['include', 'include/astrometry', 'gsl-an', 'libkd', 'qfits-an', 'util']
    include_dirs += anet + f
endforeach
include_dirs += extra + 'anet'

extra_link_args = []
if windows
    extra_link_args += ['-static-libgcc', '-static-libstdc++', '-static']
endif

py.extension_module(
    'skylib.astrometry._an_engine',
    sources: an_engine_sources,
    dependencies: [py_dep],
    limited_api: '3.10',
    link_with: libs,
    include_directories: include_directories(include_dirs),
    install: true,
    c_args: extra_link_args,
)
