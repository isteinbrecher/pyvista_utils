# -*- coding: utf-8 -*-
"""Extrude a profile along a polyline."""

# Import python modules
import pyvista as pv

# Import local stuff
from vtk_utils.polyline_cross_section import (
    polyline_cross_section as vtk_polyline_cross_section,
)


def polyline_cross_section(
    grid: pv.PolyData,
    cross_section_points,
    *,
    closed: bool = True,
    separate_surfaces: bool = False,
) -> pv.UnstructuredGrid:
    """Extrude a profile defined by the cross section coordinates along a polyline.

    This function calls the vtk version under the hood, but adds some additional
    functionality.

    Args
    ----
    grid:
        Polyline defining the centerline of the extruded structure
    cross_section_points:
        In-cross-section coordinates defining the profile
    closed:
        Flag if the profile is open or closed
    separate_surfaces:
        Per default all surface cells for a single polyline extrusion are connected.
        If this is true, then the connectivity of the returned surfaces is modified.
        The polygons at the start and end (for closed==True) will not be connected to
        the rest. For the extruded surfaces, each segment between two points in
        cross_section_points will be extruded along the centerline and not be connected
        to the bounding surfaces. This can be useful if sharp edges should be shown in
        the visualization.
    """

    cross_section_grid = pv.UnstructuredGrid(
        vtk_polyline_cross_section(grid, cross_section_points, closed=closed)
    )

    if not separate_surfaces:
        return cross_section_grid
    else:
        surface_ids = cross_section_grid.celltypes != pv.CellType.POLYGON
        grid_surfaces = cross_section_grid.extract_cells(surface_ids)
        if closed:
            n_surfaces = len(cross_section_points)
        else:
            n_surfaces = len(cross_section_points) - 1
        n_cells = grid_surfaces.n_cells
        surfaces = [
            grid_surfaces.extract_cells(range(i_start, n_cells, n_surfaces))
            for i_start in range(n_surfaces)
        ]

        if closed:
            bound_ids = cross_section_grid.celltypes == pv.CellType.POLYGON
            grid_bound = cross_section_grid.extract_cells(bound_ids)
            surfaces.append(grid_bound)

        return pv.merge(surfaces, merge_points=False)
