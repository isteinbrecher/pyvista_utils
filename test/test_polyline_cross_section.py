# -*- coding: utf-8 -*-
"""Test the functionality of polyline_cross_section"""


import os
import numpy as np
import pyvista
from pyvista_utils.clean_to_grid import clean_to_grid
from pyvista_utils.compare_grids import compare_grids
from pyvista_utils.merge_polylines import merge_polylines
from pyvista_utils.polyline_cross_section import polyline_cross_section


testing_input = os.path.join(os.path.dirname(__file__), "test_files")


def test_polyline_cross_section():
    """Test the polyline_cross_section function"""

    # Load the helix centerline
    grid = pyvista.get_reader(os.path.join(testing_input, "helix_beam.vtu")).read()
    grid = clean_to_grid(grid)
    grid = merge_polylines(grid, max_angle=np.pi)

    # Sweep a cross section along the helix
    cross_section_points = [[0, 0], [1, 0], [1, 1], [0, 1], [0.7, 0.3]]
    helix_3d = pyvista.UnstructuredGrid(
        polyline_cross_section(grid, cross_section_points)
    )

    # Compare with reference result
    helix_3d_reference = pyvista.get_reader(
        os.path.join(testing_input, "helix_cross_section.vtu")
    ).read()
    is_equal, output = compare_grids(helix_3d, helix_3d_reference, output=True)
    assert is_equal, output
