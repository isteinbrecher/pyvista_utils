# -*- coding: utf-8 -*-
"""Sort a vtk grid based on cell and/or point values."""

# Import python modules.
import pyvista as pv
import numpy as np
import vtk


def sort_grid(
    grid: pv.UnstructuredGrid, sort_point_field=None, sort_cell_field=None
) -> pv.UnstructuredGrid:
    """Sort the input grid by given arrays

    Args
    ----
    grid:
        Input grid, a sorted copy will be returned.
    sort_point_field / sort_cell_field: str, list(str)
        Data key, or list of data keys that should be used for sorting. If multiple
        keys are given, then the sorting is first performed with respect to the first
        key and so on.
    """

    def get_sorting_indices(data, n_items, sorting_keys):
        """Get the indices that shall be used for sorting the data and the reverse
        sorting indices as well

        Also process the input sorting key variable for different kinds of input
        """

        if sorting_keys is None:
            return False, None, None
        else:
            if isinstance(sorting_keys, str):
                sorting_keys = [sorting_keys]

            sort_data = np.zeros([len(sorting_keys), n_items], dtype=int)
            # Reverse the ordering of the keys, as this is required for lexsort
            for i, key in enumerate(sorting_keys[::-1]):
                sort_data[i, :] = np.array(data[key], dtype=int)
            sort_indices = np.lexsort(sort_data)
            sort_indices_reverse = np.array(range(len(sort_indices)))
            for i, index in enumerate(sort_indices):
                sort_indices_reverse[index] = i
            return True, sort_indices, sort_indices_reverse

    (
        sort_points,
        sorted_indices_points,
        sorted_indices_reverse_points,
    ) = get_sorting_indices(grid.point_data, grid.number_of_points, sort_point_field)
    (
        sort_cells,
        sorted_indices_cells,
        sorted_indices_reverse_cells,
    ) = get_sorting_indices(grid.cell_data, grid.number_of_cells, sort_cell_field)

    if not (sort_points or sort_cells):
        raise ValueError("Nothing to sort in sort_grid")

    def sort_data(data, sorted_indices):
        """Sort the given data by the given indices. If no sorting indices are given
        then the original data is returned."""
        if sorted_indices is None:
            return data
        else:
            return data[sorted_indices]

    # Get the sorted connectivity array
    cells = grid.cells
    cell_types = grid.celltypes

    # Get the sorted cells with the sorted connectivity
    points_sorted = sort_data(grid.points, sorted_indices_points)
    cell_types_sorted = sort_data(cell_types, sorted_indices_cells)
    cells_sorted_list = [None] * grid.n_cells
    index = 0
    i_cell = 0
    while index < len(cells):
        if not cell_types[i_cell] == pv.CellType.POLYHEDRON:
            n_indices = cells[index]
            sorted_connectivity = np.zeros(n_indices, dtype=int)
            for inner in range(n_indices):
                index += 1
                if sort_points:
                    sorted_connectivity[inner] = sorted_indices_reverse_points[
                        cells[index]
                    ]
                else:
                    sorted_connectivity[inner] = cells[index]
        else:
            id_vtk_list = vtk.vtkIdList()
            grid.GetFaceStream(i_cell, id_vtk_list)
            id_list = [
                id_vtk_list.GetId(i) for i in range(id_vtk_list.GetNumberOfIds())
            ]
            n_points = 0
            inner = 1
            sorted_connectivity = [id_list[0]]
            n_faces = id_list[0]
            for i_face in range(n_faces):
                n_points = id_list[inner]
                sorted_connectivity.append(n_points)
                inner += 1
                for i_point in range(n_points):
                    if sort_points:
                        sorted_connectivity.append(
                            sorted_indices_reverse_points[id_list[inner]]
                        )
                    else:
                        sorted_connectivity.append(id_list[inner])
                    inner += 1
                    n_points += 1
            index += n_points
        if sort_cells:
            cells_sorted_list[
                sorted_indices_reverse_cells[i_cell]
            ] = sorted_connectivity
        else:
            cells_sorted_list[i_cell] = sorted_connectivity
        index += 1
        i_cell += 1
    # Get the final cell connectivity array
    cells_sorted = []
    for cell in cells_sorted_list:
        cells_sorted.append(len(cell))
        cells_sorted.extend(cell)

    # Initialize the output structure
    grid_sorted = pv.UnstructuredGrid(cells_sorted, cell_types_sorted, points_sorted)

    # Sort cell and point data
    for key in grid.field_data:
        grid_sorted.field_data[key] = grid.field_data[key]
    for key in grid.cell_data:
        grid_sorted.cell_data[key] = sort_data(
            grid.cell_data[key], sorted_indices_cells
        )
    for key in grid.point_data:
        grid_sorted.point_data[key] = sort_data(
            grid.point_data[key], sorted_indices_points
        )

    return grid_sorted
