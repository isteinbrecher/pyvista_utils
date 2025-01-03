# -*- coding: utf-8 -*-
"""Test the functionality of polyline_cross_section"""

import os

import pytest
import pyvista

from pyvista_utils.polyline_cross_section import polyline_cross_section
from vtk_utils.compare_grids import compare_grids
from vtk_utils.merge_polylines import merge_polylines

from . import TESTING_INPUT


@pytest.mark.parametrize(
    ["closed", "separate_surfaces"],
    [[False, False], [False, True], [True, False], [True, True]],
)
def test_pyvista_polyline_cross_section(request, closed, separate_surfaces):
    """Test the polyline_cross_section function"""

    # Load the helix centerline
    grid = pyvista.get_reader(os.path.join(TESTING_INPUT, "helix_beam.vtu")).read()
    grid = grid.clean()
    grid = pyvista.UnstructuredGrid(merge_polylines(grid))
    grid_copy = grid.copy(deep=True)
    grid_copy.points += [2, 0, 0]
    grid = pyvista.merge([grid, grid_copy])

    # Sweep a cross section along the helix
    cross_section_points = [[0, 0], [1, 0], [1, 1], [0, 1], [0.7, 0.3]]
    helix_3d = polyline_cross_section(
        grid,
        cross_section_points,
        closed=closed,
        separate_surfaces=separate_surfaces,
    )
    helix_3d = helix_3d.connectivity()

    # Compare with reference result
    variant_name = "{}_{}".format(
        "closed" if closed else "open",
        "separate" if separate_surfaces else "connected",
    )
    test_name = request.node.name.split("test_")[1].split("[")[0] + "_" + variant_name
    helix_3d_reference = pyvista.get_reader(
        os.path.join(TESTING_INPUT, test_name + ".vtu")
    ).read()
    is_equal, output = compare_grids(helix_3d, helix_3d_reference, output=True)
    assert is_equal, output
