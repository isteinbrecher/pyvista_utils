# -*- coding: utf-8 -*-
"""Utility functions for vtk data structures"""


def vtk_id_to_list(vtk_id_list):
    """Convert a vtk id list to a python list"""
    return [
        int(vtk_id_list.GetId(i_id)) for i_id in range(vtk_id_list.GetNumberOfIds())
    ]
