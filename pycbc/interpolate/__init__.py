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

from ..scheme import schemed

@schemed("pycbc.interpolate.interpolation_")
def _interp1d_factory(*args, **kwargs):
    err_msg = "This class is a stub that should be overridden using the "
    err_msg += "scheme. You should not be seeing this error!"
    raise ValueError(err_msg)

class OneDimensionalUniformInterp(object):
    """
    Create a one-dimensional interpolation object for supported
    processing schemes. For the CPU scheme, this effectively wraps
    the scipy.interpolate.interp1d class.

    Unlike the scipy interp1d class, this class assumes uniformly
    spaced values for the independent variable. This is accomplished
    by accepting as input a single TimeSeries or FrequencySeries,
    from whose epoch and delta_t, or from its delta_f, respectively,
    the uniformly spaced independent variables are calculated.

    At present, only linear interpolation is supported. No error is
    raised if the object is called on values outside the domain of y,
    but if 'fill_value' has not been provided, those values are set
    to NaN.

    Parameters
    -----------

    y : float32 or float64
        One-dimensional TimeSeries or FrequencySeries of real values.

    fill_value : float, optional
         If provided, this value will fill in whenever interpolation
         is requested outside the domain of y. Defaults to NaN.

    Attributes:
    ------------

    fill_value : float
         The fill value for points outside the domain of y.
    """

    def __new__(cls, *args, **kwargs):
        real_cls = _interp1d_factory(*args, **kwargs)
        return real_cls(*args, **kwargs) # pylint:disable=not-callable

