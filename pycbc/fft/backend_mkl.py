#  Copyright (C) 2014 Josh Willis
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with with program; see the file COPYING. If not, write to the
#  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#  MA  02111-1307  USA

from .core import _list_available

_backend_dict = {'mkl' : 'mkl'}
_backend_list = ['mkl']

_alist, _adict = _list_available(_backend_list, _backend_dict)

mkl_backend = 'mkl'

def set_backend(backend_list):
    global mkl_backend
    for backend in backend_list:
        if backend in _alist:
            mkl_backend = backend
            break

def get_backend():
    return _adict[mkl_backend]
