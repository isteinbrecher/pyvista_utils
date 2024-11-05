# -*- coding: utf-8 -*-
"""Test the functionality of polyline_cross_section"""


import os
import pyvista
import pytest
from pyvista_utils.compare_grids import compare_grids
from pyvista_utils.merge_polylines import merge_polylines
from pyvista_utils.polyline_cross_section import polyline_cross_section


from . import TESTING_INPUT


@pytest.mark.parametrize("closed", [True, False])
def test_vtk_polyline_cross_section(request, closed):
    """Test the polyline_cross_section function"""

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
