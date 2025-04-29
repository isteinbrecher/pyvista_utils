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
"""Test the functionality of polyline_cross_section."""

import os

import pytest
import pyvista

from vtk_utils.compare_grids import compare_grids
from vtk_utils.merge_polylines import merge_polylines
from vtk_utils.polyline_cross_section import polyline_cross_section

from . import TESTING_INPUT


@pytest.mark.parametrize("closed", [True, False])
def test_vtk_polyline_cross_section(request, closed):
    """Test the polyline_cross_section function."""

    # Load the helix centerline
    grid = pyvista.get_reader(os.path.join(TESTING_INPUT, "helix_beam.vtu")).read()
    grid = grid.clean()
    grid = merge_polylines(grid)

    # Sweep a cross section along the helix
    cross_section_points = [[0, 0], [1, 0], [1, 1], [0, 1], [0.7, 0.3]]
    helix_3d = pyvista.UnstructuredGrid(
        polyline_cross_section(grid, cross_section_points, closed=closed)
    )

    # Compare with reference result
    test_name = (
        request.node.name.split("test_vtk_")[1].split("[")[0]
        + "_"
        + ("closed" if closed else "open")
    )
    helix_3d_reference = pyvista.get_reader(
        os.path.join(TESTING_INPUT, test_name + ".vtu")
    ).read()
    is_equal, output = compare_grids(helix_3d, helix_3d_reference, output=True)
    assert is_equal, output
