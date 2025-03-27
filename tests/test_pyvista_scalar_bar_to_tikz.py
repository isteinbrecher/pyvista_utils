# -*- coding: utf-8 -*-
"""Test the functionality of temporal_interpolator."""

from pathlib import Path

import pyvista as pv
from pyvista import examples

from pyvista_utils.scalar_bar_to_tikz import export_to_tikz

from . import TESTING_INPUT


def test_pyvista_temporal_interpolator(tmp_path, request):
    """Test the temporal_interpolator function."""

    testname = request.node.name

    # Load an example mesh with scalar data
    mesh = examples.load_random_hills()

    # Create a plotter object
    plotter = pv.Plotter(off_screen=True, window_size=[800, 500])

    # Add the mesh with scalar data
    plotter.add_mesh(
        mesh,
        scalars=None,
        scalar_bar_args=dict(
            title="Title",
            interactive=False,
            height=0.3,
            position_x=0.001,
            position_y=0.1,
            width=0.05,
            vertical=True,
        ),
    )

    name = "plot_to_tikz"
    export_to_tikz(tmp_path / name, plotter, dpi=250, is_testing=True)

    # Compare the created TikZ code.
    with open(tmp_path / (name + ".tex"), "r") as tikz_file:
        tikz_code = tikz_file.read().strip()
    with open(Path(TESTING_INPUT) / (testname + ".tex"), "r") as tikz_file:
        tikz_code_ref = tikz_file.read().strip()
    assert tikz_code == tikz_code_ref
