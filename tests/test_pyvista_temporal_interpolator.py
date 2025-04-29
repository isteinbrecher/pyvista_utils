# The MIT License (MIT)
#
# Copyright (c) 2023-2025 Ivo Steinbrecher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Test the functionality of temporal_interpolator."""

import os

import pyvista as pv

from pyvista_utils.temporal_interpolator import temporal_interpolator
from vtk_utils.compare_grids import compare_grids

from . import TESTING_INPUT


def test_pyvista_temporal_interpolator():
    """Test the temporal_interpolator function."""

    # Get the pvd reader
    pvd_path = os.path.join(TESTING_INPUT, "temporal_interpolator.pvd")
    pvd_reader = pv.get_reader(pvd_path)

    # Check if we can extract at a given time value
    mesh_01 = temporal_interpolator(pvd_reader, 1.0)
    mesh_01_ref = pv.get_reader(
        os.path.join(TESTING_INPUT, "temporal_interpolator_01.vtu")
    ).read()
    is_equal, output = compare_grids(mesh_01, mesh_01_ref, output=True)
    assert is_equal, output

    # Check if we can interpolate between time steps
    mesh_interpolated = temporal_interpolator(pvd_reader, 2.0)
    mesh_interpolated_ref = pv.get_reader(
        os.path.join(TESTING_INPUT, "temporal_interpolator_interpolated.vtu")
    ).read()
    is_equal, output = compare_grids(
        mesh_interpolated, mesh_interpolated_ref, output=True
    )
    assert is_equal, output
