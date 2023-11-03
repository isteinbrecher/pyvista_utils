# -*- coding: utf-8 -*-
"""Compare two grids to each other"""

# Import python modules.
import numpy as np
from vtk.util import numpy_support as vtk_numpy_support


def compare_grids(grid_1, grid_2, tol=1e-10, output=False) -> bool:
    """Compare two grids to each other

    Args
    ----
    grid_1, grid_2:
        Input grids which are compared to each other
    tol: float
        Tolerance for comparing floating point data
    output: bool
        If output of the results of the comparison should be given
    """

    def compare_arrays(array_1, array_2, name):
        """Compare two arrays"""

        if array_1 is None and array_2 is None:
            return True, f"{name}: OK (empty)"
        elif array_1 is None or array_2 is None:
            return (
                False,
                f"{name}: Array 1 is {type(array_1)}, array 2 is {type(array_2)}",
            )

        n_1 = array_1.GetNumberOfTuples()
        n_2 = array_2.GetNumberOfTuples()
        if not n_1 == n_2:
            return (
                False,
                f"{name}: Number of tuples does not match, got {n_1} and {n_2}",
            )

        n_1 = array_1.GetNumberOfComponents()
        n_2 = array_2.GetNumberOfComponents()
        if not n_1 == n_2:
            return (
                False,
                f"{name}: Number of components does not match, got {n_1} and {n_2}",
            )

        t_1 = array_1.GetDataTypeAsString()
        t_2 = array_2.GetDataTypeAsString()
        if not t_1 == t_2:
            return (
                False,
                f"{name}: Data type does not match, got {t_1} and {t_2}",
            )

        if not np.allclose(
            vtk_numpy_support.vtk_to_numpy(array_1),
            vtk_numpy_support.vtk_to_numpy(array_1),
            atol=tol,
        ):
            return (
                False,
                f"{name}: Data values do not match",
            )

        return True, f"{name}: Compares OK"

    return_value = True
    lines = []

    # Compare the point coordinates
    compare_value, string = compare_arrays(
        grid_1.GetPoints().GetData(),
        grid_2.GetPoints().GetData(),
        "point_coordinates",
    )
    return_value = compare_value and return_value
    lines.append(string)

    # Compare the cells
    compare_value, string = compare_arrays(
        grid_1.GetCellTypesArray(), grid_2.GetCellTypesArray(), "cell_types"
    )
    return_value = compare_value and return_value
    lines.append(string)

    compare_value, string = compare_arrays(
        grid_1.GetCells().GetData(), grid_2.GetCells().GetData(), "cell_connectivity"
    )
    return_value = compare_value and return_value
    lines.append(string)

    compare_value, string = compare_arrays(
        grid_1.GetFaces(), grid_2.GetFaces(), "face_connectivity"
    )
    return_value = compare_value and return_value
    lines.append(string)

    def compare_data_fields(data_1, data_2, name):
        """Compare multiple data sets grouped together"""

        names_1, names_2 = [
            set([data.GetArrayName(i) for i in range(data.GetNumberOfArrays())])
            for data in [data_1, data_2]
        ]

        if not names_1 == names_2:
            return (
                False,
                [f"{name}: Data fields do not match, got {names_1} and {names_2}"],
            )

        if len(names_1) == 0:
            return True, [f"{name}: OK (empty)"]

        lines = []
        return_value = True
        for field_name in names_1:
            compare_value, string = compare_arrays(
                data_1.GetArray(field_name),
                data_2.GetArray(field_name),
                f"{name}::{field_name}",
            )
            return_value = compare_value and return_value
            lines.append(string)

        return return_value, lines

    # Compare actual data
    compare_value, strings = compare_data_fields(
        grid_1.GetFieldData(), grid_2.GetFieldData(), "field_data"
    )
    return_value = compare_value and return_value
    lines.extend(strings)
    compare_value, strings = compare_data_fields(
        grid_1.GetCellData(), grid_2.GetCellData(), "cell_data"
    )
    return_value = compare_value and return_value
    lines.extend(strings)
    compare_value, strings = compare_data_fields(
        grid_1.GetPointData(), grid_2.GetPointData(), "point_data"
    )
    return_value = compare_value and return_value
    lines.extend(strings)

    if output:
        return return_value, lines
    else:
        return return_value
