# -*- coding: utf-8 -*-
"""Test the functionality of merge_polylines"""

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
