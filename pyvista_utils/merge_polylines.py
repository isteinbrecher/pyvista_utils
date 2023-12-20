# -*- coding: utf-8 -*-
"""Merge polylines with each other that represent a continuous curve"""


# Import python modules.
import numpy as np
import vtk

# Import local stuff
from .utils import vtk_id_to_list


def merge_polylines(
    grid: vtk.vtkUnstructuredGrid,
    *,
    output_grid: vtk.vtkUnstructuredGrid = None,
    max_angle=0.5 * np.pi,
) -> vtk.vtkUnstructuredGrid:
    """Merge polylines with each other that represent a continuous curve

    Args
    ----
    grid:
        Input grid, can only contain polylines. In order for continuous segments
        to be found, the polylines have to connect to the same points via the
        connectivity matrix.
    output_grid:
        If this is given, the merged grid will be stores in this grid. Otherwise,
        a new grid will be returned.
    max_angle: float
        Threshold for maximum angle between successive segments along a continuous
        polyline.
    """

    if not (output_grid_given := (output_grid is not None)):
        output_grid = vtk.vtkUnstructuredGrid()
    output_grid.Initialize()

    # Add the points to the output.
    output_grid.SetPoints(grid.GetPoints())
    for i in range(grid.GetPointData().GetNumberOfArrays()):
        output_grid.GetPointData().AddArray(grid.GetPointData().GetArray(i))

    # Check that all cells are poly lines.
    n_cells = grid.GetNumberOfCells()
    for i in range(n_cells):
        if not grid.GetCellType(i) == 4:
            raise ValueError("Only poly lines (vtk type 4) are supported")

    def find_connected_cells(grid, cell_id):
        """Start with polyline "cell_id" in grid and search all polylines connected
        to that one. Also return all points of those lines in order for the combined
        polyline.
        """

        def add_cell_recursive(connected_cell_points, old_cells, initial_point_id):
            """Start at the initial point and loop over polylines as long as a connectivity is found."""
            grid.GetPointCells(initial_point_id, id_list)
            cell_connectivity = vtk_id_to_list(id_list)
            new_cell_ids = [
                cell_id for cell_id in cell_connectivity if cell_id not in old_cells
            ]
            if len(cell_connectivity) > 2 or len(new_cell_ids) == 0:
                # In this case we either have a bifurcation point or are at the end
                # the poly line.
                return connected_cell_points, old_cells

            # Add this cell and its points (in correct order).
            new_cell_id = new_cell_ids[0]
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

            # Check the angel of the corner and decide whether or not to merge
            # this.
            if extend:
                corner_point_ids = connected_cell_points[-2:] + [new_cell_point_ids[1]]
            else:
                corner_point_ids = [new_cell_point_ids[-2]] + connected_cell_points[:2]
            points = [np.array(grid.GetPoint(j)) for j in corner_point_ids]
            vec_1 = points[1] - points[0]
            vec_1 = vec_1 / np.linalg.norm(vec_1)
            vec_2 = points[2] - points[1]
            vec_2 = vec_2 / np.linalg.norm(vec_2)
            dot = np.dot(vec_1, vec_2)
            if 1.0 <= dot and dot < 1.0 + 1e-12:
                dot = 1.0
            angle = np.arccos(dot)
            if np.abs(angle) > max_angle:
                # Angle between beam elements is more than the maximum angle,
                # which indicates an edge and not a continuous polyline.
                return connected_cell_points, old_cells

            # Extend the merged poly line points.
            if extend:
                connected_cell_points.extend(new_cell_point_ids[1:])
                next_start_index = -1
            else:
                connected_cell_points = new_cell_point_ids[:-1] + connected_cell_points
                next_start_index = 0

            old_cells.append(new_cell_id)
            connected_cell_points, old_cells = add_cell_recursive(
                connected_cell_points, old_cells, new_cell_point_ids[next_start_index]
            )

            return connected_cell_points, old_cells

        id_list = vtk.vtkIdList()
        connected_cell_points = vtk_id_to_list(grid.GetCell(cell_id).GetPointIds())
        old_cells = [cell_id]

        start_id = connected_cell_points[0]
        end_id = connected_cell_points[-1]
        connected_cell_points, old_cells = add_cell_recursive(
            connected_cell_points, old_cells, start_id
        )
        connected_cell_points, old_cells = add_cell_recursive(
            connected_cell_points, old_cells, end_id
        )

        return old_cells, connected_cell_points

    # Start with the first poly line and search all poly lines connected to that
    # line and so on. Then do the same with the first poly line that was not found
    # and so on.
    cell_ids = list(range(n_cells))
    new_cells = []
    while True:
        # Take the next available cell and look for all connected cells.
        for i in cell_ids:
            if i is not None:
                next_id = i
                break
        else:
            break
        old_cells, connected_cell_points = find_connected_cells(grid, next_id)

        # Mark the found cell IDs.
        for found_cell_id in old_cells:
            cell_ids[found_cell_id] = None

        # Create the poly line for this beam.
        new_cell = vtk.vtkPolyLine()
        new_cell.GetPointIds().SetNumberOfIds(len(connected_cell_points))
        for i, index in enumerate(connected_cell_points):
            new_cell.GetPointIds().SetId(i, index)
        new_cells.append(new_cell)

    # Add all new cells to the output data.
    output_grid.Allocate(len(new_cells), 1)
    for cell in new_cells:
        output_grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

    if not output_grid_given:
        return output_grid
