#!/usr/bin/env python
import sys
import os
from glob import glob
from numpy.distutils.core import Extension, setup

# A hack: compiler override for all commands, incl. those that do not explicitly
# support the --compiler option
for arg in sys.argv[1:]:
    if arg.find('--compiler=') == 0:
        from distutils import ccompiler
        defcomp = list(getattr(ccompiler, '_default_compilers'))
        for i, spec in enumerate(defcomp):
            if spec[0] == os.name:
                defcomp[i] = (os.name, arg[11:])
                del sys.argv[sys.argv.index(arg)]
                ccompiler._default_compilers = tuple(defcomp)
                break

tparty = 'skylib/thirdparty/'
anet = tparty + 'astrometry.net/'
extra = tparty + 'anet_extra/'
gsl = anet + 'gsl-an/'

an_engine_ext = Extension(
    name='skylib.astrometry._an_engine',
    sources=['skylib/astrometry/an_engine_wrap.c'] +
    [anet + 'util/{}.c'.format(fn) for fn in (
        'an-endian', 'bl', 'codekd', 'datalog', 'errors', 'fit-wcs', 'fitsbin',
        'fitsfile', 'fitsioutils', 'fitstable', 'gslutils', 'healpix', 'index',
        'ioutils', 'log', 'mathutil', 'os-features', 'permutedsort', 'quadfile',
        'sip', 'sip-utils', 'starkd', 'starutil', 'starxy', 'tic',)] +
    [anet + 'blind/{}.c'.format(fn) for fn in(
        'matchobj', 'quad-utils', 'solver', 'tweak', 'tweak2', 'verify',)],
    libraries=[
        ('gsl-an', dict(
            sources=glob(gsl + 'blas/*') +
            [gsl + 'linalg/{}.c'.format(fn)
             for fn in ('bidiag', 'cholesky', 'householder', 'lu', 'qr',
                        'svd')] +
            [gsl + 'multiroots/{}.c'.format(f)
             for f in ('broyden', 'convergence', 'dnewton', 'fdfsolver',
                       'fdjac', 'fsolver', 'gnewton', 'hybrid',
                       'hybridj', 'newton')] +
            glob(gsl + 'sys/*.c') + glob(gsl + 'cblas/*.c') +
            glob(gsl + 'err/*.c') +
            [gsl + 'block/{}.c'.format(fn) for fn in ('block', 'init')] +
            [gsl + 'matrix/{}.c'.format(f)
             for f in ('copy', 'init', 'matrix', 'rowcol', 'submatrix',
                       'swap', 'view')] +
            [gsl + 'vector/{}.c'.format(f)
             for f in ('copy', 'init', 'oper', 'prop', 'subvector', 'swap',
                       'vector')] +
            [gsl + 'permutation/{}.c'.format(f)
             for f in ('init', 'permutation', 'permute')],
            include_dirs=[gsl for fn in ('', 'cblas')] + [extra + 'gsl-an'],
        )),
        ('kd', dict(
            sources=[
                anet + 'libkd/{}.c'.format(f)
                for f in (
                    'dualtree', 'dualtree_nearestneighbour',
                    'dualtree_rangesearch', 'kdint_ddd', 'kdint_dds',
                    'kdint_ddu', 'kdint_dss', 'kdint_duu',
                    'kdint_fff', 'kdtree', 'kdtree_dim',
                    'kdtree_fits_io', 'kdtree_mem')],
            include_dirs=[anet + fn
                          for fn in ('include', 'include/astrometry',
                                     'qfits-an', 'util')],
            macros=[('DONT_INCLUDE_OS_FEATURES_CONFIG_H', None)],
        )),
        ('qfits-an', dict(
            sources=glob(anet + 'qfits-an/*.c'),
            include_dirs=[anet + fn
                          for fn in ('include', 'include/astrometry', 'util')],
        )),
    ],
    include_dirs=[anet + fn
                  for fn in ('include', 'include/astrometry', 'gsl-an',
                             'libkd', 'qfits-an', 'util')],
    # Don't use os-features-config.h, define OS-specific macros in setup.py
    define_macros=[('DONT_INCLUDE_OS_FEATURES_CONFIG_H', None)],
)

if sys.platform.startswith('win'):
    # Astrometry.net on Win needs external implementations of mmap and regex
    for libname, libdef in an_engine_ext.libraries:
        if libname in ('kd', 'qfits-an'):
            libdef['include_dirs'] += [extra, extra + 'regex']
    an_engine_ext.libraries += [
        ('posix', dict(
            sources=glob(extra + 'sys/*.c'),
            include_dirs=[extra + 'sys'],
        )),
        ('regex', dict(
            sources=glob(extra + 'regex/*.c'),
            include_dirs=[extra + 'regex'],
        )),
        'iconv',
    ]
    an_engine_ext.include_dirs += [extra + 'regex']
    an_engine_ext.define_macros += [
        ('_POSIX', None),
        ('NEED_CANONICALIZE_FILE_NAME', '1'),
        ('NEED_DECLARE_QSORT_R', '1'),
        ('NEED_QSORT_R', '1'),
        ('NEED_SWAP_QSORT_R', '0'),
    ]
elif sys.platform.startswith('freebsd') or sys.platform.startswith('darwin'):
    an_engine_ext.define_macros += [
        ('NEED_CANONICALIZE_FILE_NAME', '1'),
        ('NEED_DECLARE_QSORT_R', '1'),
        ('NEED_QSORT_R', '0'),
        ('NEED_SWAP_QSORT_R', '0'),
    ]
else:
    an_engine_ext.define_macros += [
        ('NEED_CANONICALIZE_FILE_NAME', '0'),
        ('NEED_DECLARE_QSORT_R', '0'),
        ('NEED_QSORT_R', '0'),
        ('NEED_SWAP_QSORT_R', '1'),
    ]

setup(
    name='SkyLib',
    version='0.1.2',
    requires=['numpy', 'astropy(>=1.2)', 'scipy(>=1.0)', 'sep', 'astroscrappy'],
    packages=[
        'skylib', 'skylib.astrometry', 'skylib.calibration', 'skylib.combine',
        'skylib.extraction', 'skylib.io', 'skylib.photometry',
        'skylib.sonification', 'skylib.util'],
    ext_modules=[an_engine_ext],
    scripts=['scripts/fits2wav.py'],
)
