# Copyright (C) 2020 Joshua L. Willis
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import numpy as np
import mako.template
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv
from ..types import TimeSeries, FrequencySeries
from ..scheme import mgr as _scheme_mgr

kernel_sources = mako.template.Template("""
#include <pycuda-helpers.hpp>
#include <math_constants.h>

texture<${textype}, 1, cudaReadModeElementType> val_tex;

__global__ void uniform_linear_interp(${valtype} *output, const double* locs,
                                      double delta, double start, int N, int M,
                                      ${valtype} fill, unsigned int use_fill){

    ${valtype} hi, lo;
    double diff, locv, loci;
    int j;

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N){
      locv = locs[i];
      loci = (locv-start)/delta;
      j = __float2int_rd(loci);
      if ((j >= 0) && (j < M-1)){
        lo = ${texref_func}(val_tex, j);
        hi = ${texref_func}(val_tex, j+1);
        diff = loci-j;
        output[i] = lo + (hi-lo)*diff;
      // Allow for possibility that point is exactly at last sample point
      } else if (locv == (start+M*delta)) {
        output[i] = ${texref_func}(val_tex, M-1);
      } else {
         if use_fill {
            output[i] = fill;
         } else {
            output[i] = ${cuda_nan};
         }
      }
    }

    return;

}
"""

single_module = SourceModule(
                    kernel_sources.render(textype="float",
                        valtype="float", texref_func="tex1Dfetch",
                        cuda_nan="CUDART_NAN_F"))        

double_module = SourceModule(
                    kernel_sources.render(textype="fp_tex_double",
                        valtype="double", texref_func="fp_tex1Dfetch",
                        cuda_nan="CUDART_NAN"))        

single_vals = drv.pagelocked_empty((4096), np.float32, mem_flags=drv.host_alloc_flags.DEVICEMAP)
s_vptr = np.intp(val.base.get_device_pointer())

double_vals = drv.pagelocked_empty((4096), np.float64, mem_flags=drv.host_alloc_flags.DEVICEMAP)
d_vptr = np.intp(val.base.get_device_pointer())

class OneDimensionalUniformInterpolationCUDA(object):
    def __init__(self, y, fill_value=None):
 
       if isinstance(y, TimeSeries):
            self._startv = np.float64(y._epoch)
            self._delta = np.float64(y._delta_t)
        elif isinstance(y, FrequencySeries):
            self._startv = np.float64(0.0)
            self._delta = np.float64(y._delta_f)
        else:
            raise ValueError("Input y must be a PyCBC TimeSeries or FrequencySeries")

        self._ary = y

        if fill_value is None:
            self._use_fill = np.uint32(0)
            self._fill_val = 0.0
            self.fill_value = np.nan
        else:
            self._use_fill = np.uint32(1)
            self._fill_val = fill_value
            self.fill_value = fill_value

        if self._ary.precision is 'single':
            self._fill_val = np.float32(self._fill_val)
            self._mod = single_module
            self._texref = self._mod.get_texref("val_tex")
            self._ary.data.bind_to_texref_ext(self._texref, allow_offset=False)
            self._fn = self._mod.get_function("uniform_linear_interp")
            self._fn.prepare("PPDDiiFi", texrefs=[self._texref])
            self._out_ptr = s_vptr
            self._base_out = single_vals
        else:
            self._fill_val = np.float64(self._fill_val)
            self._mod = double_module
            self._texref = self._mod.get_texref("val_tex")
            self._ary.data.bind_to_texref_ext(self._texref, allow_double_hack=True,
                                              allow_offset=False)
            self._fn = self._mod.get_function("uniform_linear_interp")
            self._fn.prepare("PPDDiiDi", texrefs=[self._texref])
            self._out_ptr = d_vptr
            self._base_out = double_vals

        self._M = np.uint32(len(self._ary))


    def __call__(self, x):
        N = np.unint32(len(x))
        locs = gpuarray.to_gpu(x).gpudata
        self._fn.prepared_call((nb,1), (nt, 1, 1), self._out_ptr, locs,
                               self._delta, self._startv, N, self._M,
                               self._fill_val, self._use_fill)
        scheme_mgr.state.context.synchronize()
        return self._base_out[0:N].copy()


def _interp1d_factory(*args, **kwargs):
    return OneDimensionalUniformInterpolationCUDA


