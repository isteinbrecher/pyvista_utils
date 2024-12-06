# -*- coding: utf-8 -*-
"""Interpolate between time steps in pvd time step series"""


import numpy as np
import pyvista as pv


def temporal_interpolator(
    pvd_collection: pv.PVDReader, time: float, *, tol=1e-10
) -> pv.UnstructuredGrid:
    """Interpolate between time steps in pvd time step series

    Args
    ----
    pvd_collection:
        PVD reader containing the pvd collection file
    time:
        Time value for the desired interpolated data
    tol:
        Tolerance for deciding if a time step is exactly found
    """

    def get_mesh(time_index) -> pv.UnstructuredGrid:
        pvd_collection.set_active_time_point(time_index)
        return pvd_collection.read()[0]

    time_values = np.array(pvd_collection.time_values)

    # Check if we hit the time step exactly
    diff = time_values - time
    abs_diff = np.abs(diff)
    closest_time_step = np.argmin(abs_diff)
    if abs_diff[closest_time_step] < tol:
        return get_mesh(closest_time_step)

    positive_indices = np.where(diff > 0)[0]
    end_index = positive_indices[np.argmin(diff[positive_indices])]
    start_index = end_index - 1
    start = time_values[start_index]
    end = time_values[end_index]
    alpha = (time - start) / (end - start)
    factor = [1 - alpha, alpha]

    output_mesh = get_mesh(start_index)
    end_mesh = get_mesh(end_index)

    output_mesh.points = factor[0] * output_mesh.points + factor[1] * end_mesh.points
    for key in output_mesh.cell_data.keys():
        output_mesh.cell_data[key] = (
            factor[0] * output_mesh.cell_data[key] + factor[1] * end_mesh.cell_data[key]
        )
    for key in output_mesh.point_data.keys():
        output_mesh.point_data[key] = (
            factor[0] * output_mesh.point_data[key]
            + factor[1] * end_mesh.point_data[key]
        )
    return output_mesh
