# -*- coding: utf-8 -*-
"""Merge lines or polylines with each other that represent a continuous
curve."""

import numpy as np
import vtk

from vtk_utils.vtk_data_structures_utils import vtk_id_to_list


def merge_polylines(
    grid: vtk.vtkUnstructuredGrid,
    *,
    output_grid: vtk.vtkUnstructuredGrid = None,
    smooth_angle=135.0 * np.pi / 180.0,
) -> vtk.vtkUnstructuredGrid:
    """Merge lines or polylines with each other that represent a continuous
    curve.

    Args
    ----
    grid:
        Input grid, can only contain lines or polylines. In order for continuous
        segments to be found, the cells have to connect to the same points via
        the connectivity matrix.
    output_grid:
        If this is given, the merged grid will be stores in this grid. Otherwise,
        a new grid will be returned.
    smooth_angle: float
        Threshold for maximum angle between successive segments along a continuous
        line.
    """

    if not (output_grid_given := (output_grid is not None)):
        output_grid = vtk.vtkUnstructuredGrid()
    output_grid.Initialize()

    # Add the points (and the point data) to the output.
    output_grid.SetPoints(grid.GetPoints())
    for i in range(grid.GetPointData().GetNumberOfArrays()):
        output_grid.GetPointData().AddArray(grid.GetPointData().GetArray(i))

    # Check that all cells are lines or polylines.
    n_cells = grid.GetNumberOfCells()
    for i in range(n_cells):
        cell = grid.GetCell(i)
        if not (isinstance(cell, vtk.vtkLine) or isinstance(cell, vtk.vtkPolyLine)):
            raise ValueError(
                "Only lines (vtk type 3) and poly lines (vtk type 4) are supported. Got {} (vtk type {})".format(
                    type(cell), grid.GetCellType(i)
                )
            )

    def get_angle_between_lines(tangent_at_point, point_id, cell_id):
        """Get the dot product of the tangents at the connection point between
        connecting cells."""

        cell_point_ids = vtk_id_to_list(grid.GetCell(cell_id).GetPointIds())
        if cell_point_ids[-1] == point_id:
            cell_tangent_point_indices = [-2, -1]
        elif cell_point_ids[0] == point_id:
            cell_tangent_point_indices = [1, 0]
        else:
            raise ValueError("Given point index does not match the connectivity array")
        points_for_tangent = [
            np.array(grid.GetPoint(cell_point_ids[index]))
            for index in cell_tangent_point_indices
        ]
        cell_tangent = points_for_tangent[1] - points_for_tangent[0]
        cell_tangent = cell_tangent / np.linalg.norm(cell_tangent)

        # Get the angle between the two
        return np.dot(tangent_at_point, cell_tangent)

    def find_next_connected_polyline(grid, old_cell_tracker):
        """Start with the first old cell that was not found yet. Then search
        all cells connected to that one.

        Return all point ids that make up the new poly line. Return None
        if all cells have been found.
        """

        # Take the next available cell and look for all connected cells
        for i in old_cell_tracker:
            if i is not None:
                next_cell_id = i
                break
        else:
            return None

        def add_next_cell(connected_cell_points, old_cell_tracker, connected_point_id):
            """Start at the initial point and loop over lines as long as a
            connectivity is found."""
            id_list = vtk.vtkIdList()
            grid.GetPointCells(connected_point_id, id_list)
            cell_connectivity = vtk_id_to_list(id_list)
            possible_next_cell_ids = [
                cell_id
                for cell_id in cell_connectivity
                if old_cell_tracker[cell_id] is not None
            ]

            if len(possible_next_cell_ids) == 0:
                # In this case we are at the end of the poly line
                return connected_cell_points, None

            # Get the the outward facing tangent at the given point
            if connected_cell_points[-1] == connected_point_id:
                tangent_point_indices = [-2, -1]
            elif connected_cell_points[0] == connected_point_id:
                tangent_point_indices = [1, 0]
            else:
                raise ValueError(
                    "Given point index does not match the connectivity array"
                )
            points_for_tangent = [
                np.array(grid.GetPoint(connected_cell_points[index]))
                for index in tangent_point_indices
            ]
            tangent = points_for_tangent[1] - points_for_tangent[0]
            tangent = tangent / np.linalg.norm(tangent)

            # Check the angle between this line and all connected cells
            smooth_connected_cells = []
            for cell_id in cell_connectivity:
                dot = get_angle_between_lines(tangent, connected_point_id, cell_id)
                if np.cos(smooth_angle) > dot:
                    smooth_connected_cells.append(cell_id)

            if len(smooth_connected_cells) == 0 or len(smooth_connected_cells) > 1:
                # In this case there are either no or multiple lines connected which are
                # smooth to the given one. There is no unique way to continue this, so we
                # stop here.
                return connected_cell_points, None
            elif old_cell_tracker[smooth_connected_cells[0]] is None:
                # The smooth connected cell is already accounted for in the new cells
                return connected_cell_points, None
            else:
                # We want to continue along the found smooth cell
                new_cell_id = smooth_connected_cells[0]

            # Add the new cell and its points (in correct order).
            new_cell_point_ids = vtk_id_to_list(grid.GetCell(new_cell_id).GetPointIds())
            if new_cell_point_ids[0] == connected_cell_points[-1]:
                # First point of this cell is added to the last point of the last
                # cell.
                extend = True
            elif new_cell_point_ids[-1] == connected_cell_points[-1]:
                # Last point of this cell is added to the last point of the last
                # cell.
                extend = True
                new_cell_point_ids.reverse()
            elif new_cell_point_ids[0] == connected_cell_points[0]:
                # First point of this cell is added to the first point of the last
                # cell.
                extend = False
                new_cell_point_ids.reverse()
            elif new_cell_point_ids[-1] == connected_cell_points[0]:
                # Last point of this cell is added to the first point of the last
                # cell.
                extend = False
            else:
                raise ValueError("This should not happen")

            # Extend the merged poly line points
            if extend:
                connected_cell_points.extend(new_cell_point_ids[1:])
                next_start_index = -1
            else:
                connected_cell_points = new_cell_point_ids[:-1] + connected_cell_points
                next_start_index = 0

            old_cell_tracker[new_cell_id] = None
            return connected_cell_points, new_cell_point_ids[next_start_index]

        old_cell_tracker[next_cell_id] = None
        connected_cell_points = vtk_id_to_list(grid.GetCell(next_cell_id).GetPointIds())
        start_id = connected_cell_points[0]
        end_id = connected_cell_points[-1]

        for start_index in [start_id, end_id]:
            next_point_id = start_index
            while next_point_id is not None:
                connected_cell_points, next_point_id = add_next_cell(
                    connected_cell_points, old_cell_tracker, next_point_id
                )

        return connected_cell_points

    # Start with the first cell and search all cells connected to that cell and so on.
    # Then do the same with the first cell that was not found and so on.
    old_cell_tracker = list(range(n_cells))
    new_cells = []
    while True:
        connected_cell_point_ids = find_next_connected_polyline(grid, old_cell_tracker)

        if connected_cell_point_ids is None:
            break
        else:
            # Create the found poly line
            new_cell = vtk.vtkPolyLine()
            new_cell.GetPointIds().SetNumberOfIds(len(connected_cell_point_ids))
            for i, index in enumerate(connected_cell_point_ids):
                new_cell.GetPointIds().SetId(i, index)
            new_cells.append(new_cell)

    # Add all new cells to the output data
    output_grid.Allocate(len(new_cells), 1)
    for cell in new_cells:
        output_grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

    if not output_grid_given:
        return output_grid
