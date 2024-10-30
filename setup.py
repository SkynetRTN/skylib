#!/usr/bin/env python
import sys
import os
from glob import glob
from distutils import ccompiler as cc
import numpy.distutils.misc_util


linux_platform = sys.platform.startswith('linux')
bsd_platform = 'bsd' in sys.platform
osx_platform = 'darwin' in sys.platform
win_platform = sys.platform.startswith('win')


# A hack: compiler override for all commands, incl. those that do not
# explicitly support the --compiler option
ccompiler = None
for arg in sys.argv[1:]:
    if arg.startswith('--compiler='):
        defcomp = list(getattr(cc, '_default_compilers'))
        val = arg.split('=', 1)[1]
        for i, compspec in enumerate(defcomp):
            if compspec[0] == os.name:
                defcomp[i] = (os.name, val)
                sys.argv.remove(arg)
                cc._default_compilers = tuple(defcomp)
                ccompiler = val
                break

# Obtain C compiler ID if unspecified
if ccompiler is None:
    ccompiler = cc.get_default_compiler()

# Prevent linking against MSVCRxx.dll on Windows when compiled with mingw32
if win_platform:
    class IntWithLstrip(int):
        def lstrip(self, _=None):
            return self

    dummy_msvc_runtime_library = False
    save_msvc_runtime_version = numpy.distutils.misc_util.msvc_runtime_version

    def msvc_runtime_version_override():
        if dummy_msvc_runtime_library:
            return
        return save_msvc_runtime_version()

    numpy.distutils.misc_util.msvc_runtime_version = \
        msvc_runtime_version_override


# Astrometry.net won't compile with MSVC; force Mingw-W64
an_engine_compiler = 'mingw32'
if win_platform and ccompiler != an_engine_compiler:
    from numpy.distutils.command import build_clib, build_ext

    old_build_clib = build_clib.build_clib

    # noinspection PyClassicStyleClass
    class BuildClibMingw32(old_build_clib):
        def build_a_library(self, build_info, libname, libraries):
            if libname in ('gsl-an', 'kd', 'qfits-an', 'posix', 'regex'):
                save_ccompiler = self.compiler
                self.compiler = cc.new_compiler(
                    compiler=an_engine_compiler, verbose=self.verbose,
                    dry_run=self.dry_run, force=self.force)
                self.compiler.customize(self.distribution)
                self.compiler.customize_cmd(self)
                self.compiler.show_customization()
                old_build_clib.build_a_library(
                    self, build_info, libname, libraries)
                # noinspection PyAttributeOutsideInit
                self.compiler = save_ccompiler
            else:
                old_build_clib.build_a_library(
                    self, build_info, libname, libraries)
    build_clib.build_clib = BuildClibMingw32

    old_build_ext = build_ext.build_ext

    # noinspection PyClassicStyleClass
    class BuildExtMingw32(old_build_ext):
        def build_extension(self, ext):
            global dummy_msvc_runtime_library
            if ext.name == 'skylib.astrometry._an_engine':
                prev_msvc_runtime_library = dummy_msvc_runtime_library
                dummy_msvc_runtime_library = True
                save_ccompiler = self.compiler
                self.compiler = cc.new_compiler(
                    compiler=an_engine_compiler, verbose=self.verbose,
                    dry_run=self.dry_run, force=self.force)
                self.compiler.customize(self.distribution)
                self.compiler.customize_cmd(self)
                self.compiler.show_customization()
                old_build_ext.build_extension(self, ext)
                # noinspection PyAttributeOutsideInit
                self.compiler = save_ccompiler
                dummy_msvc_runtime_library = prev_msvc_runtime_library
            else:
                old_build_ext.build_extension(self, ext)
    build_ext.build_ext = BuildExtMingw32


# Link with static standard libraries on mingw and cygwin to simplify binary
# distribution
extra_link_args = []
mingw32_link_args = ['-static-libgcc', '-static-libstdc++', '-static']
if ccompiler == 'mingw32':
    extra_link_args += mingw32_link_args

    # Prevent linking against MSVCRxx.dll
    dummy_msvc_runtime_library = True


tparty = 'skylib/thirdparty/'
anet = tparty + 'astrometry.net/'
extra = tparty + 'anet_extra/'
gsl = anet + 'gsl-an/'


# Define platform-dependent extensions
import numpy.distutils.core
Extension = numpy.distutils.core.Extension

# noinspection PyTypeChecker
an_engine_ext = Extension(
    name='skylib.astrometry._an_engine',
    sources=['skylib/astrometry/an_engine_wrap.c'] +
    [anet + 'util/{}.c'.format(fn) for fn in (
        'an-endian', 'bl', 'codekd', 'datalog', 'errors', 'fit-wcs', 'fitsbin',
        'fitsfile', 'fitsioutils', 'fitstable', 'gslutils', 'healpix', 'index',
        'ioutils', 'log', 'matchobj', 'mathutil', 'permutedsort', 'quadfile',
        'sip', 'sip-utils', 'starkd', 'starutil', 'starxy', 'tic',
    )] +
    [anet + 'solver/{}.c'.format(fn) for fn in (
        'quad-utils', 'solver', 'tweak', 'tweak2', 'verify',
    )],
    libraries=[
        ('gsl-an', dict(
            sources=glob(gsl + 'blas/*') +
            [gsl + 'linalg/{}.c'.format(fn)
             for fn in ('bidiag', 'cholesky', 'householder', 'lu', 'qr',
                        'svd')] +
            [gsl + 'multiroots/{}.c'.format(f)
             for f in ('broyden', 'convergence', 'dnewton', 'fdfsolver',
                       'fdjac', 'fsolver', 'gnewton', 'hybrid', 'hybridj',
                       'newton')] +
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
                    'kdint_ddu', 'kdint_dss', 'kdint_duu', 'kdint_fff',
                    'kdint_lll', 'kdtree', 'kdtree_dim', 'kdtree_fits_io',
                    'kdtree_mem')],
            include_dirs=[anet + fn
                          for fn in ('include', 'include/astrometry',
                                     'qfits-an', 'util')] + [extra + 'anet'],
        )),
        ('qfits-an', dict(
            sources=glob(anet + 'qfits-an/*.c'),
            include_dirs=[anet + fn
                          for fn in ('include', 'include/astrometry',
                                     'qfits-an', 'util')],
        )),
    ],
    include_dirs=[anet + fn
                  for fn in ('include', 'include/astrometry', 'gsl-an',
                             'libkd', 'qfits-an', 'util')] + [extra + 'anet'],
    extra_link_args=extra_link_args,
)

if win_platform:
    # Fix index file corruption in anqfits.c on Windows
    for _libname, _libdef in an_engine_ext.libraries:
        if _libname == 'qfits-an':
            for f in ('anqfits', 'qfits_memory'):
                _libdef['sources'].remove(anet + 'qfits-an\\' + f + '.c')
                _libdef['sources'].append(extra + 'qfits-an/' + f + '.c')
    for f in ('ioutils',):
        an_engine_ext.sources.remove(anet + 'util/{}.c'.format(f))
        an_engine_ext.sources.append(extra + '{}.c'.format(f))

if ccompiler == 'mingw32':
    # mingw-w64 needs external implementations of certain features
    for _libname, libdef in an_engine_ext.libraries:
        if _libname in ('kd', 'qfits-an'):
            libdef['include_dirs'] += [extra, extra + 'regex']
        if _libname == 'qfits-an':
            # Fake defs missing in mingw-w64
            libdef['include_dirs'] += [extra, extra + 'qfits-an',
                                       extra + 'endian']
            libdef.setdefault('extra_compiler_args', [])
            libdef['extra_compiler_args'] += [
                '-include', extra + 'qfits-an/qfits-an-sys-stat.h',
                '-D_POSIX_THREAD_SAFE_FUNCTIONS',  # No localtime_r()
            ]
    an_engine_ext.libraries += [
        ('posix', dict(
            sources=glob(extra + 'sys/*.c') + [extra + 'an-defs.c'],
            include_dirs=[extra + 'sys'],
            macros=[('_POSIX', None)],
        )),
        ('regex', dict(
            sources=glob(extra + 'regex/*.c'),
            include_dirs=[extra + 'regex'],
        )),
        'iconv', 'ws2_32',
    ]
    an_engine_ext.include_dirs += [extra, extra + 'regex', extra + 'endian']
    an_engine_ext.extra_compile_args += ['-include', extra + 'an-defs.h']
    an_engine_ext.define_macros += [('_POSIX', None)]
elif ccompiler == 'cygwin':
    # Cygwin does not define __int64
    an_engine_ext.define_macros += [('__int64', 'long long')]

numpy.distutils.core.setup(
    name='SkyLib',
    version='0.2.0',
    requires=['numpy(>=1.18)', 'astropy(>=1.2)', 'scipy(>=1.0)', 'sep(==1.0.3)',
              'astroscrappy(==1.0.8)'],
    provides='skylib',
    packages=[
        'skylib', 'skylib.astrometry', 'skylib.calibration', 'skylib.combine',
        'skylib.enhancement', 'skylib.extraction', 'skylib.io',
        'skylib.photometry', 'skylib.sonification', 'skylib.util'],
    ext_modules=[an_engine_ext],
    package_data={
        'skylib.sonification': ['*.wav'],
        'skylib.astrometry': ['ngc2000.dat'],
    },
    scripts=['scripts/fits2wav.py'],
)
