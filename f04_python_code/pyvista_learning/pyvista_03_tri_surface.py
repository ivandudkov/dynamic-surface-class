# PyVista tutorial
# https://www.youtube.com/watch?v=FmNmRBsEBHE&ab_channel=SoftwareUnderground
# https://github.com/banesullivan/transform-2021

import numpy as np
import pyvista as pv

# This script will be about how to create a triangulated surface

# 1. Define a simple Gaussian surface
n = 20
x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)

xx, yy = np.meshgrid(x, y)
A, b = 100, 100

# Define Z-values for Gaussian surface
zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

# 1.1 Get the points as a 2D numpy array (N by 3)
points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]

# 2. Now, use those points create a point cloud PyVista data object: pv.PolyData
# simply passing points to the PolyData constructor
cloud = pv.PolyData(points)

# 2.1 Plot that object
cloud.plot(point_size=15)

# 3. PyVista has a bunch of methods for triangulating points to create a surface
# That way we will perform a triangulation to turn discrete points into a
# connected surface by mesh.delaunay_2d() triangulation
surf = cloud.delaunay_2d()

# 3.1 Plot the surface
surf.plot(show_edges=True)

# 4. Masked Triangulations
# There is also a way to do masked triangulations. To show that let's define
# a new point cloud:
x = np.arange(10, dtype=float)
xx, yy, zz = np.meshgrid(x, x, [0])
points = np.column_stack((xx.ravel(order='F'),
                          yy.ravel(order='F'),
                          zz.ravel(order='F')))
# Perturb the points
points[:, 0] += np.random.rand(len(points)) * 0.3
points[:, 1] += np.random.rand(len(points)) * 0.3

# Create the point cloud mesh to triangulate from the coordinates
cloud = pv.PolyData(points)

# 4.1 Plot the point cloud
cloud.plot(cpos='xy', show_edges=True)

# 4.2 Run the triangulation for these points
surf = cloud.delaunay_2d(alpha=1.0)
# here we've introduced 'alpha' parameter - we need it, if some of our outer edges
# are unconstrained and the triangulation added unwanted triangles

surf.plot(cpos='xy', show_edges=True)

# 5. Now, to perform the masked triangulation, we should add a polygon
# to ignore during the triangulation via the 'edge_source' parameter
# 5.1 Define a polygonal hole with a clockwise polygon:
ids = [22, 23, 24, 25, 35, 45, 44, 43, 42, 32]

# 5.2 Create a PolyData to store the boundary
polygon = pv.PolyData()

# 5.3 Make sure it has the same points as the mesh being triangulated
polygon.points = points

# 5.4 But only has faces in regions to ignore
polygon.faces = np.array([len(ids), ] + ids)

# 5.5 Perform triangulation
surf = cloud.delaunay_2d(alpha=1.0, edge_source=polygon)

# 5.6 Plot the data
p = pv.Plotter()
p.add_mesh(surf, show_edges=True)
p.add_mesh(polygon, color='red', opacity=0.5)
p.show(cpos='xy')
















