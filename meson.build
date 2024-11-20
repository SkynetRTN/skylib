project(
    'skylib',
    'c',
    version: run_command(
        'python',
        ['-c', 'import sys; sys.path = ["."]; from skylib import __version__; print(__version__, end="")'],
        check: true
    ).stdout().strip(),
    meson_version: '>=1.6.0',
    default_options: [
        'buildtype=release',
        'c_std=gnu99',
        'warning_level=3',
    ]
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

tparty = 'skylib/thirdparty/'
anet = tparty + 'astrometry.net/'
extra = tparty + 'anet_extra/'
gsl = anet + 'gsl-an/'

add_project_arguments('-DNPY_NO_DEPRECATED_API=0', language : 'c')
add_project_arguments('-DNPY_TARGET_VERSION=NPY_1_26_API_VERSION', language : 'c')

windows = host_machine.system() == 'windows'

if windows
    add_project_link_arguments(['-lucrt', '-static-libgcc', '-static', '-liconv', '-lws2_32'], language: 'c')
    add_project_arguments(['-include', extra + 'an-defs.h', '-D_POSIX', '-D_POSIX_C_SOURCE'], language: 'c')
endif

foreach p: ['skylib', 'skylib/astrometry', 'skylib/calibration', 'skylib/color', 'skylib/combine', 'skylib/enhancement',
            'skylib/extraction', 'skylib/io', 'skylib/photometry', 'skylib/sonification', 'skylib/util']
    sources = run_command(
        py, ['-c', 'from glob import glob; print("\\n".join(glob("' + p + '/*.py")), end="")'], check: true
    ).stdout().split('\n')
    py.install_sources(sources, subdir: p)
endforeach

py.install_sources(['skylib/astrometry/ngc2000.dat'], subdir: 'skylib/astrometry')
py.install_sources(['skylib/sonification/start.wav', 'skylib/sonification/stop.wav'], subdir: 'skylib/sonification')
py.install_sources(['scripts/fits2wav.py'], subdir: 'skylib/scripts')

# Astrometry.net source files
an_engine_sources = ['skylib/astrometry/an_engine_wrap.c']
foreach fn: [
        'an-endian', 'bl', 'codekd', 'datalog', 'errors', 'fit-wcs', 'fitsbin', 'fitsfile', 'fitsioutils', 'fitstable',
        'gslutils', 'healpix', 'index', 'log', 'matchobj', 'mathutil', 'permutedsort', 'quadfile', 'sip', 'sip-utils',
        'starkd', 'starutil', 'starxy', 'tic',
        ]
    an_engine_sources += anet + 'util/' + fn + '.c'
endforeach
if windows
    an_engine_sources += extra + 'ioutils.c'
else
    an_engine_sources += anet + 'util/ioutils.c'
endif

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
if windows
    kd_include_dirs += [extra, extra + 'regex']
endif

qfits_an_sources = run_command(
    py,
    ['-c', 'from glob import glob; print("\\n".join(glob("' + anet + 'qfits-an/*.c")), end="")'],
    check: true).stdout().split('\n')
qfits_an_include_dirs = []
foreach fn: ['include', 'include/astrometry', 'qfits-an', 'util']
    qfits_an_include_dirs += anet + fn
endforeach
qfits_an_c_args = []
if windows
    # Fix index file corruption in anqfits.c on Windows
    sources = []
    foreach f: qfits_an_sources
        if not (f.contains('anqfits.c') or f.contains('qfits_memory.c'))
            sources += f
        endif
    endforeach
    qfits_an_sources = sources + [extra + 'qfits-an/anqfits.c', extra + 'qfits-an/qfits_memory.c']

    qfits_an_include_dirs += [extra, extra + 'regex', extra + 'qfits-an', extra + 'endian']
    qfits_an_c_args += [
        '-include', extra + 'qfits-an/qfits-an-sys-stat.h',
        '-D_POSIX_THREAD_SAFE_FUNCTIONS',  # No localtime_r()
    ]
endif

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
        c_args: qfits_an_c_args,
    ),
]
if windows
    posix_sources = run_command(
        py,
        ['-c', 'from glob import glob; print("\\n".join(glob("' + extra + 'sys/*.c")), end="")'],
        check: true).stdout().split('\n')
    posix_sources += extra + 'an-defs.c'
    regex_sources = run_command(
        py,
        ['-c', 'from glob import glob; print("\\n".join(glob("' + extra + 'regex/*.c")), end="")'],
        check: true
    ).stdout().split('\n')
    libs += [
        static_library(
            'posix',
            posix_sources,
            include_directories: include_directories(extra + 'sys'),
            c_args: ['-D_POSIX'],
        ),
        static_library(
            'regex',
            regex_sources,
            include_directories: include_directories(extra + 'regex'),
            c_args: ['-D_GNU_SOURCE'],
        ),
    ]
endif

# Global include directories
np_incdir = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
include_dirs = [np_incdir]
foreach f: ['include', 'include/astrometry', 'gsl-an', 'libkd', 'qfits-an', 'util']
    include_dirs += anet + f
endforeach
include_dirs += extra + 'anet'

if windows
    cc = meson.get_compiler('c')
    assert(cc.get_id() == 'gcc', 'Astrometry.net extension can only be compiled with mingw-w64 on Windows')

    include_dirs += [extra, extra + 'regex', extra + 'endian']
endif

an_engine = py.extension_module(
    '_an_engine',
    sources: an_engine_sources,
    dependencies: [py_dep],
    limited_api: '3.10',
    link_with: libs,
    include_directories: include_directories(include_dirs),
    install: true,
    subdir: 'skylib/astrometry',
)
