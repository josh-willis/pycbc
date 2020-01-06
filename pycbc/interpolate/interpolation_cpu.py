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

from scipy.interpolate import interp1d
from ..types import TimeSeries, FrequencySeries

class OneDimensionalUniformInterpCPU(object):
    def __init__(self, y, fill_value=None):
        if isinstance(y, TimeSeries):
            self._interp_obj = interp1d(y.sample_times, y, kind='linear',
                                        bounds_error=False,
                                        fill_value=fill_value,
                                        assume_sorted=True)
        elif isinstance(y, FrequencySeries):
            self._interp_obj = interp1d(y.sample_frequencies, y,
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value=fill_value,
                                        assume_sorted=True)
        else:
            raise ValueError("Input y must be a PyCBC TimeSeries or FrequencySeries")

        self.fill_value=fill_value

        def __call__(self, *args, **kwargs):
            return self._interp_obj(*args, **kwargs)

def _interp1d_factory(*args, **kwargs):
    return OneDimensionalUniformInterpCPU


