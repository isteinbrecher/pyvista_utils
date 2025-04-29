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
"""Test the functionality of merge_polylines."""

import os

import pyvista

from vtk_utils.compare_grids import compare_grids
from vtk_utils.merge_polylines import merge_polylines

from . import TESTING_INPUT


def test_vtk_merge_polylines():
    """Test the merge_polylines function."""

    grid = pyvista.get_reader(
        os.path.join(TESTING_INPUT, "merge_polylines_raw.vtu")
    ).read()
    grid = grid.clean()
    grid_merged = merge_polylines(grid)
    grid_ref = pyvista.get_reader(
        os.path.join(TESTING_INPUT, "merge_polylines_reference.vtu")
    ).read()

    compare = compare_grids(grid_merged, grid_ref, output=True)
    assert compare[0], compare[1]
