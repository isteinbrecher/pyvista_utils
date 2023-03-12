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
    unique_points = points[unique_indices]
    n_unique_points = len(unique_indices)

    unique_id_count = np.zeros(n_unique_points, dtype=int)
    for unique_id in inverse_indices:
        unique_id_count[unique_id] += 1
    unique_id_count_inverse = 1.0 / unique_id_count

    # Map the cell indices to the unique points
    cells_out = mesh_in.cells.copy()
    celltypes_out = mesh_in.celltypes.copy()
    index = 0
    while index < len(cells_out):
        number_of_points_connected_to_cell = cells_out[index]
        my_slice = slice(index + 1, index + number_of_points_connected_to_cell + 1)
        cells_out[my_slice] = [inverse_indices[id] for id in cells_out[my_slice]]
        index += number_of_points_connected_to_cell + 1

    # Create the output mesh
    mesh_out = pyvista.UnstructuredGrid(cells_out, celltypes_out, unique_points)

    # Set the data values
    mesh_out.cell_data.update(mesh_in.cell_data)
    for point_data_name, point_data_in in mesh_in.point_data.items():
        point_data_out = np.zeros(point_data_in.shape)[:n_unique_points]
        for i_original, i_unique in enumerate(inverse_indices):
            # Todo move the scaling out of the loop and use a np function to do it
            point_data_out[i_unique] += (
                point_data_in[i_original] * unique_id_count_inverse[i_unique]
            )
        mesh_out.point_data[point_data_name] = point_data_out

    return mesh_out
