/* RCRLib.i */

%module RCRLib
%include <std_vector.i>

namespace std {
 %template(DubVec) vector<double>;
 %template(BoolVec) vector<bool>;

}

%{
#include "RCR.h"
%}

%include "RCR.h"
