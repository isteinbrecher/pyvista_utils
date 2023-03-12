import numpy as np
import pyvista

from meshpy.geometric_search import find_close_points, point_partners_to_unique_indices


def clean_to_grid(mesh_in, **kwargs):

    # Find unique points
    points = mesh_in.points
    partners, n_partners = find_close_points(points, **kwargs)
    unique_indices, inverse_indices = point_partners_to_unique_indices(
        partners, n_partners
    )
    unique_id_count = np.zeros(len(unique_indices), dtype=int)
    for unique_id in inverse_indices:
        unique_id_count[unique_id] += 1
    unique_points = points[unique_indices]

    # Map the cell indices to the unique points
    cells_out = mesh_in.cells.copy()
    celltypes_out = mesh_in.celltypes.copy()
    index = 0
    while index < len(cells_out):
        number_of_points_connected_to_cell = cells_out[index]
        my_slice = slice(index + 1, index + number_of_points_connected_to_cell + 1)
        cells_out[my_slice] = [inverse_indices[id] for id in cells_out[my_slice]]
        index += number_of_points_connected_to_cell + 1

    mesh_out = pyvista.UnstructuredGrid(cells_out, celltypes_out, unique_points)

    return mesh_out
