# -*- coding: utf-8 -*-
"""Test the functionality of pyvista_utlis"""

import os
import numpy as np
import unittest
import pyvista
from pyvista_utils.clean_to_grid import clean_to_grid
from pyvista_utils.compare_grids import compare_grids
from pyvista_utils.merge_polylines import merge_polylines
from pyvista_utils.sort_grid import sort_grid


testing_input = os.path.join(os.path.dirname(__file__), "test_files")


class TestSortGrid(unittest.TestCase):
    """This class contains test cases for the sort grid function"""

    def test_sort_grid_complete(self):
        """Test the sort grid function. This is done by comparing FEM grids that were
        generated with a different number of processors. In this test we completely
        sort the grid, i.e., we sort all points such that their ordering is completely
        defined by the sorting keys"""

        mesh_serial, mesh_parallel = [
            pyvista.get_reader(os.path.join(testing_input, name)).read()
            for name in ["sort_serial.vtu", "sort_parallel.vtu"]
        ]

        # Since each element writes it's own nodes, we first sort the nodes by the cell id
        mesh_serial = mesh_serial.cell_data_to_point_data(pass_cell_data=True)
        mesh_parallel = mesh_parallel.cell_data_to_point_data(pass_cell_data=True)

        # To make the sorting more challenging we split the cell and point IDs into individual
        # fields representing the digits
        parallel_node_gid = np.array(mesh_parallel.point_data["node_gid"], dtype=int)
        mesh_parallel.point_data["node_gid_001"] = parallel_node_gid % 10
        mesh_parallel.point_data["node_gid_010"] = parallel_node_gid % 100
        mesh_parallel.point_data["node_gid_100"] = parallel_node_gid % 1000
        parallel_cell_gid = np.array(mesh_parallel.cell_data["element_gid"], dtype=int)
        mesh_parallel.cell_data["element_gid_01"] = parallel_cell_gid % 10
        mesh_parallel.cell_data["element_gid_10"] = parallel_cell_gid % 100
        mesh_parallel_sorted = sort_grid(
            mesh_parallel,
            sort_point_field=[
                "element_gid",
                "node_gid_100",
                "node_gid_010",
                "node_gid_001",
            ],
            sort_cell_field=["element_gid_10", "element_gid_01"],
        )

        # The serial array also has to be sorted, because we also sort with respect to the
        # node ids above.
        mesh_serial_sorted = sort_grid(
            mesh_serial,
            sort_point_field=["element_gid", "node_gid"],
            sort_cell_field=["element_gid"],
        )

        # Remove the arbitrary created data
        del mesh_parallel_sorted.point_data["node_gid_001"]
        del mesh_parallel_sorted.point_data["node_gid_010"]
        del mesh_parallel_sorted.point_data["node_gid_100"]
        del mesh_parallel_sorted.cell_data["element_gid_01"]
        del mesh_parallel_sorted.cell_data["element_gid_10"]

        # Compare the two grids
        compare = compare_grids(mesh_parallel_sorted, mesh_serial_sorted, output=True)
        self.assertTrue(compare[0], msg=compare[1])

    def test_sort_grid_partial(self):
        """Test the sort grid function. This is done by comparing FEM grids that were
        generated with a different number of processors. In this test we completely
        only partially sort the grid, i.e., this tests how the original ordering
        is preserved after a sort."""

        mesh_serial, mesh_parallel = [
            pyvista.get_reader(os.path.join(testing_input, name)).read()
            for name in ["sort_serial.vtu", "sort_parallel.vtu"]
        ]

        # Since each element writes it's own nodes, we first sort the nodes by the cell id
        mesh_parallel = mesh_parallel.cell_data_to_point_data(pass_cell_data=True)

        # Sort the parallel grid w.r.t. to the cell id
        mesh_parallel_sorted = sort_grid(
            mesh_parallel, sort_point_field=["element_gid"]
        )

        # Remove data created for sorting
        del mesh_parallel_sorted.point_data["element_gid"]

        # Compare the two grids
        compare = compare_grids(mesh_parallel_sorted, mesh_serial, output=True)
        self.assertTrue(compare[0], msg=compare[1])

    def test_sort_grid_mixed_types(self):
        """Test that the sort grid function can handle polyhedrons."""

        mesh_mixed_cells = pyvista.read(
            os.path.join(testing_input, "mixed_cell_types.vtu")
        )

        # Sort the parallel grid
        mesh_mixed_cells.point_data["sort_id"] = range(
            mesh_mixed_cells.number_of_points
        )[::-1]
        mesh_mixed_cells_sorted = sort_grid(
            mesh_mixed_cells, sort_point_field=["sort_id"]
        )

        # Compare with the reference grid
        mesh_mixed_cells_reference = pyvista.read(
            os.path.join(testing_input, "mixed_cell_types_sorted.vtu")
        )
        compare = compare_grids(
            mesh_mixed_cells_sorted, mesh_mixed_cells_reference, output=True
        )
        self.assertTrue(compare[0], msg=compare[1])


class TestMergePolylines(unittest.TestCase):
    """This class contains test cases for the sort merge_polylines function"""

    def test_merge_polylines(self):
        """Test the merge_polylines function."""

        grid = pyvista.get_reader(
            os.path.join(testing_input, "merge_polylines_raw.vtu")
        ).read()
        grid = clean_to_grid(grid)
        grid_merged = merge_polylines(grid, max_angle=np.pi)
        grid_ref = pyvista.get_reader(
            os.path.join(testing_input, "merge_polylines_reference.vtu")
        ).read()
        compare = compare_grids(grid_merged, grid_ref, output=True)
        self.assertTrue(compare[0], msg=compare[1])


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
