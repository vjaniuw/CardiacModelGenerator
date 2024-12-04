#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:32:54 2024

@author: vinayjani
"""

import unittest
import numpy as np
import pyvista as pv
import wx
from CardiacModelGenerator import (
    CardiacMeshalyzer,
    generate_point_cloud,
    generate_tetra_mesh,
    clean_tetra_mesh,
    get_cell_quality,
)

class TestCardiacModelGenerator(unittest.TestCase):

    def test_generate_point_cloud(self):

        coords1 = np.random.rand(100, 3)
        masks1 = np.ones((100,))
  
        point_cloud = generate_point_cloud(coords1=coords1, masks1=masks1)

        self.assertIsNotNone(point_cloud)
        self.assertGreater(len(point_cloud.points), 0)
    
    def test_generate_tetra_mesh(self):

        point_cloud = np.random.rand(100, 3)
        point_cloud = pv.PolyData(point_cloud)

        tetra_mesh = generate_tetra_mesh(point_cloud_cleaned=point_cloud)

        self.assertIsNotNone(tetra_mesh)
        self.assertGreater(len(tetra_mesh.cells), 0)
    
    def test_clean_tetra_mesh(self):

        grid = np.random.rand(100, 3)
        grid = pv.PolyData(grid)
        grid = grid.delaunay_3d()

        cleaned_mesh = clean_tetra_mesh(grid, subdivisions=2, poisson_iterations=3)

        self.assertIsNotNone(cleaned_mesh)
        self.assertGreater(len(cleaned_mesh.cells), 0)
    
    def test_get_cell_quality(self):

        mesh = np.random.rand(100, 3)
        mesh = pv.PolyData(mesh)
        mesh= mesh.delaunay_3d()
        # Call function
        quality = get_cell_quality(final_volumetric_mesh=mesh)
        # Assertions
        self.assertIsInstance(quality, 'pyvista.core.pointset.PolyData')
    


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    

