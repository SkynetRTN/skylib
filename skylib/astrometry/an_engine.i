
%module(threads="1") an_engine

%include <typemaps.i>


%feature("autodoc", "1");

%{
#define NPY_NO_DEPRECATED_API 8
#include <numpy/arrayobject.h>
#include "astrometry/index.h"
#include "astrometry/starxy.h"
#include "astrometry/sip.h"
#include "astrometry/matchobj.h"
#include "astrometry/solver.h"
#include "astrometry/fit-wcs.h"
#include "astrometry/tweak2.h"
#include "astrometry/healpix.h"
#include "astrometry/log.h"
%}

%init %{
import_array();
%}


// Handle arrays (e.g. crpix)

%include "carrays.i"
%include "cpointer.i"
%array_class(double, DoubleArray);
%array_class(int, IntArray);
%array_class(unsigned int, UIntArray);
%pointer_cast(void *, double *, voidp_to_doublep);
%pointer_cast(void *, int *, voidp_to_intp);
%pointer_cast(void *, unsigned int *, voidp_to_uintp);


// Handle int* and double* output arguments

%typemap(argout) double * {
    PyObject *o, *o2, *o3;
    o = SWIG_From_double(*$1);
    if ((!$result) || ($result == Py_None)) {
        $result = o;
    } else {
        if (!PyTuple_Check($result)) {
            PyObject *o2 = $result;
            $result = PyTuple_New(1);
            PyTuple_SetItem($result,0,o2);
        }
        o3 = PyTuple_New(1);
        PyTuple_SetItem(o3,0,o);
        o2 = $result;
        $result = PySequence_Concat(o2,o3);
        Py_DECREF(o2);
        Py_DECREF(o3);
    }
}

%typemap(in,numinputs=0) double * (double temp) {
    $1 = &temp;
}


%typemap(argout) int * {
    PyObject *o, *o2, *o3;
    o = SWIG_From_int(*$1);
    if ((!$result) || ($result == Py_None)) {
        $result = o;
    } else {
        if (!PyTuple_Check($result)) {
            PyObject *o2 = $result;
            $result = PyTuple_New(1);
            PyTuple_SetItem($result,0,o2);
        }
        o3 = PyTuple_New(1);
        PyTuple_SetItem(o3,0,o);
        o2 = $result;
        $result = PySequence_Concat(o2,o3);
        Py_DECREF(o2);
        Py_DECREF(o3);
    }
}

%typemap(in,numinputs=0) int * (int temp) {
    $1 = &temp;
}


// Use NumPy arrays in starxy_set_*_array()

%typemap(in) const double * ( PyArrayObject *arr = NULL ) {
    if ($input != Py_None)
    {
	arr = (PyArrayObject *)PyArray_ContiguousFromAny($input, NPY_DOUBLE, 1, 1);
	if (!arr) return NULL;
	$1 = (double *)PyArray_DATA(arr);
    }
    else
    {
	$1 = (double *)NULL;
    }
}

%typemap(argout) const double * {}

%typemap(freearg) const double * {
    Py_XDECREF(arr$argnum);
}


//TODO: Wrap starxy_copy_*() and starxy_to_*()


// starxy_get() returns an xy pair

%typemap(argout) double *xy {
    PyObject *o, *o2, *o3;
    o = PyTuple_New(2);
    PyTuple_SetItem(o, 0, SWIG_From_double($1[0]));
    PyTuple_SetItem(o, 1, SWIG_From_double($1[1]));
    if ((!$result) || ($result == Py_None)) {
        $result = o;
    } else {
        if (!PyTuple_Check($result)) {
            PyObject *o2 = $result;
            $result = PyTuple_New(1);
            PyTuple_SetItem($result,0,o2);
        }
        o3 = PyTuple_New(1);
        PyTuple_SetItem(o3,0,o);
        o2 = $result;
        $result = PySequence_Concat(o2,o3);
        Py_DECREF(o2);
        Py_DECREF(o3);
    }
}

%typemap(in,numinputs=0) double *xy (double temp[2]) {
    $1 = temp;
}


// In/out tan_t and sip_t

%typemap(in) sip_t *sip (void *argp = 0, int res = 0) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, $disown | %convertptr_flags);
    if (!SWIG_IsOK(res)) {
	%argument_fail(res, "$type", $symname, $argnum);
    }
    $1 = %reinterpret_cast(argp, $ltype);
}

%typemap(argout) sip_t *sip {
    $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), SWIGTYPE_p_sip_t, 0);
}


%typemap(in) tan_t *wcstan (void *argp = 0, int res = 0) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, $disown | %convertptr_flags);
    if (!SWIG_IsOK(res)) {
	%argument_fail(res, "$type", $symname, $argnum);
    }
    $1 = %reinterpret_cast(argp, $ltype);
}

%typemap(argout) tan_t *wcstan {
    $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), SWIGTYPE_p_tan_t, 0);
}


// Wrap the basic engine functionality

%define WarnUnusedResult %enddef
%include "astrometry/an-bool.h"
%include "astrometry/index.h"
%include "astrometry/qfits_table.h"
%include "astrometry/fitstable.h"
%include "astrometry/starxy.h"
%include "astrometry/starkd.h"
%include "astrometry/sip.h"
%include "astrometry/fit-wcs.h"
%include "astrometry/matchobj.h"
%include "astrometry/solver.h"
%include "astrometry/tweak2.h"


// Misc. extra functions to wrap

%apply (double *xy) {(double *closestradec)}

double healpix_distance_to_radec(int hp, int Nside, double ra, double dec,
    double* closestradec);
int healpix_within_range_of_radec(int hp, int Nside, double ra, double dec,
                                  double radius);

void log_init(int);
