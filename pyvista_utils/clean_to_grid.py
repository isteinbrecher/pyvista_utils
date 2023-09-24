# -*- coding: utf-8 -*-
"""Implement a clean_to_grid functionality similar to the one provided in
ParaView. This allows to clean UnstructuredGrid (the clean command in 
pyvista only works for PolyData).
"""

import numpy as np
import pyvista
from scipy.spatial import KDTree


def clean_to_grid(
    mesh_in: pyvista.UnstructuredGrid, *, tol: float = 1e-8
) -> pyvista.UnstructuredGrid:
    """Clean the mesh, i.e., overlapping points are replaced by a single point.
    Point data at overlapping points is averaged. Cell data is kept as it is,
    as only the cell connectivity changes with this filter but not the cell
    itself.

    This function is inspired by:
    https://gist.github.com/gilrrei/e72bc7dd75eb3a3ad38173d0dbbb9c98

    Args
    ----
    mesh_in:
        Mesh to be cleaned
    tol: float
        Tolerance to be used in the closest point search
    Return
    ----
    mesh_out:
        The cleaned mesh (a copy of the original mesh where double points are merged)
    """

    # Find points that are close to each other
    points = mesh_in.points
    kd_tree = KDTree(points)
    pairs = kd_tree.query_pairs(r=tol, output_type="ndarray")

    # Group together all point ids of points that are at the same spatial dimension
    point_id_connectivity = [{i} for i in range(len(points))]
    for pair in pairs:
        point_id_connectivity[pair[0]].add(pair[1])
        point_id_connectivity[pair[1]].add(pair[0])
    point_id_connectivity = frozenset(
        [frozenset(item) for item in point_id_connectivity]
    )

    # Create a mapping from the original ids to the new (spatially) unique ids.
    # Also get the new point coordinates and count how many old points a new point
    # "unifies".
    n_cleaned_points = len(point_id_connectivity)
    point_id_mapping_old_to_new = [None for i in range(len(points))]
    cleaned_points = np.zeros((n_cleaned_points, 3), dtype=points.dtype)
    cleaned_points_count = np.zeros(n_cleaned_points, dtype=int)
    for unique_index, item in enumerate(point_id_connectivity):
        for index in item:
            if point_id_mapping_old_to_new[index] is None:
                point_id_mapping_old_to_new[index] = unique_index
                cleaned_points[unique_index] = points[index]
                cleaned_points_count[unique_index] += 1
            else:
                raise ValueError(
                    "There are connected clusters of points. This can happen if the tolerance is to high."
                )
    cleaned_points_count_inverse = 1.0 / cleaned_points_count

    # Get the new cell connectivity
    cells_out = mesh_in.cells.copy()
    celltypes_out = mesh_in.celltypes.copy()
    index = 0
    while index < len(cells_out):
        number_of_points_connected_to_cell = cells_out[index]
        my_slice = slice(index + 1, index + number_of_points_connected_to_cell + 1)
        cells_out[my_slice] = [
            point_id_mapping_old_to_new[id] for id in cells_out[my_slice]
        ]
        index += number_of_points_connected_to_cell + 1

    # Create the output mesh
    mesh_out = pyvista.UnstructuredGrid(cells_out, celltypes_out, cleaned_points)

    # Set the data values
    mesh_out.cell_data.update(mesh_in.cell_data)
    for point_data_name, point_data_in in mesh_in.point_data.items():
        point_data_out = np.zeros((n_cleaned_points, 3))
        for i_original, i_unique in enumerate(point_id_mapping_old_to_new):
            point_data_out[i_unique] += point_data_in[i_original]
        point_data_out *= cleaned_points_count_inverse[:, np.newaxis]
        mesh_out.point_data[point_data_name] = point_data_out

    return mesh_out
