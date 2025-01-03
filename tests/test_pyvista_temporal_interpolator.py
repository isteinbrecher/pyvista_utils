# -*- coding: utf-8 -*-
"""Test the functionality of temporal_interpolator"""

import os

import pyvista as pv

from pyvista_utils.temporal_interpolator import temporal_interpolator
from vtk_utils.compare_grids import compare_grids

from . import TESTING_INPUT


def test_pyvista_temporal_interpolator():
    """Test the temporal_interpolator function"""

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
