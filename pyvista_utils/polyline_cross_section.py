# -*- coding: utf-8 -*-
"""Extrude a profile along a polyline."""

# Import python modules.
import numpy as np
import vtk
from vtk.util import numpy_support as vtk_numpy_support


def vtk_id_to_list(vtk_id_list):
    """Convert a vtk id list to a python list. TODO: Move this to a general utils file."""
    return [
        int(vtk_id_list.GetId(i_id)) for i_id in range(vtk_id_list.GetNumberOfIds())
    ]


def polyline_cross_section(
    grid: vtk.vtkUnstructuredGrid, polygon_points
) -> vtk.vtkUnstructuredGrid:
    """Extrude a profile defined by the corner points of a polygon along a polyline."""

    # Get information about input grid
    n_cells = grid.GetNumberOfCells()

    # Get information about polygon
    n_points_polygon = len(polygon_points)

    # Check that all cells are poly lines.
    for i in range(n_cells):
        if not grid.GetCellType(i) == 4:
            raise ValueError("Only poly lines (vtk type 4) are supported")

    # Data arrays
    point_data = grid.GetPointData()
    base_vector_data = [None] * 3
    data = {}
    for i_point_data in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i_point_data)
        if name.startswith("base_vector_"):
            i_base = int(name.split("_")[-1])
            base_vector_data[i_base - 1] = vtk_numpy_support.vtk_to_numpy(
                point_data.GetArray(i_point_data)
            )
        else:
            data[name] = point_data.GetArray(i_point_data)

    # New points
    new_point_coordinates = vtk.vtkPoints()
    new_polygons = []
    new_quad4 = []

    def extrude_cross_section_polyline(polyline: vtk.vtkPolyLine):
        """Extrude the cross section along the given polyline"""

        i_start = new_point_coordinates.GetNumberOfPoints()

        point_ids = vtk_id_to_list(polyline.GetPointIds())
        for i, point_id in enumerate(point_ids):
            i_start_inner = new_point_coordinates.GetNumberOfPoints()

            coordinates = np.array(grid.GetPoint(point_id))
            base_vectors = [base_vector_data[i_dir][point_id] for i_dir in range(3)]
            for i_polygon, p_polygon in enumerate(polygon_points):
                new_coordinate = (
                    coordinates
                    + p_polygon[0] * base_vectors[1]
                    + p_polygon[1] * base_vectors[2]
                )
                new_point_coordinates.InsertNextPoint(new_coordinate)

                # Set the quad4 cells
                if not i == 0:
                    new_cell = vtk.vtkQuad()
                    new_cell.GetPointIds().SetNumberOfIds(4)
                    new_cell.GetPointIds().SetId(
                        0, i_start_inner - n_points_polygon + i_polygon
                    )
                    new_cell.GetPointIds().SetId(1, i_start_inner + i_polygon)
                    if i_polygon == n_points_polygon - 1:
                        i_polygon = -1
                    new_cell.GetPointIds().SetId(2, i_start_inner + i_polygon + 1)
                    new_cell.GetPointIds().SetId(
                        3, i_start_inner - n_points_polygon + i_polygon + 1
                    )
                    new_quad4.append(new_cell)

        i_end = new_point_coordinates.GetNumberOfPoints() - 1

        # Set the front and end polygon
        for index, factor in [[i_start, 1], [i_end, -1]]:
            new_cell = vtk.vtkPolygon()
            new_cell.GetPointIds().SetNumberOfIds(n_points_polygon)
            for i_point_polygon in range(n_points_polygon):
                new_cell.GetPointIds().SetId(
                    i_point_polygon, index + factor * i_point_polygon
                )
            new_polygons.append(new_cell)

    # Create the new data for each polyline
    for i_cell in range(n_cells):
        extrude_cross_section_polyline(grid.GetCell(i_cell))

    # Add the new points and cells
    output_grid = vtk.vtkUnstructuredGrid()
    output_grid.Initialize()
    output_grid.SetPoints(new_point_coordinates)
    output_grid.Allocate(len(new_polygons) + len(new_quad4), 1)
    for cell in new_polygons + new_quad4:
        output_grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

    return output_grid
