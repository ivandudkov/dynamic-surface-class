# PyVista tutorial
# https://www.youtube.com/watch?v=FmNmRBsEBHE&ab_channel=SoftwareUnderground
# https://github.com/banesullivan/transform-2021

import numpy as np
import pyvista as pv

# How to create a point cloud
# In PyVista point cloud is a mesh object that contains only nodes

# 1. Create some nodes XYZ
nodes = np.random.rand(100, 3) * 10
# 2. Create mesh, containing only nodes
mesh = pv.PolyData(nodes)
# 3. Visualize point cloud
# mesh.plot(render_points_as_spheres=True)

# 4. To add colors that represents Z-values, let's do next trick
# for any mesh you can define... some specific attribute using
# python dictionary stuff. Any mesh has two structures that attributes
# can be assigned for: nodes and cells. Dictionaries attaching can be
# called by
# 1) meshname.point_arrays['something'] = X # data X should have same length as nodes array of that mesh
# 2) meshname.cell_arrays['something'] = Y # data Y should have same length as cells numbers of that mesh

# Let's add attribute 'z_points'
mesh.point_arrays['z_points'] = nodes[:, 2]

# We also can assign attribute ignoring call-function
mesh['z_points'] = nodes[:, 2]


# 5. Now, let's plot it again, but in plot function
# The plot function will colorize our points automatically
# by the first founded attribute
mesh.plot(render_points_as_spheres=True)

# 6. Let's access to mesh's metadata:
# Number of Mesh's cells
print('Number of cells: %d' % mesh.n_cells)
# Number of Mesh's points
print('Number of points: %d' % mesh.n_points)
# Number of Mesh's arrays (attributes)
print('Number of arrays: %d' % mesh.n_arrays)
# Mesh bounds - Spatial bounds of your mesh:
# [X min, X max, Y min, Y max, Z min, Z max]
print('Mesh bounds are:')
print(mesh.bounds)
# Coordinates of the center of this mesh:
print('Mesh center is:')
print(mesh.center)
# Let's print basic info about our mesh:
print(mesh)
# Let's plot mesh's array names
print('Array names:')
print(mesh.array_names)

# 7. Access to mesh's data
# Access to points:
print(mesh.points)
# Type of the mesh array is "pyvista_ndarray", which is just
# a wrapping of a numpy array => you can treat to it
# as numpy array:
print(type(mesh.points))
# To be sure, let's check it:
print('Does pyvista_ndarray is an instance of the np.ndarray?')
print(isinstance(mesh.points, np.ndarray))
# Let's find array's min value:
min = mesh.points.min()
print('Minimum value of the mesh points is %f' % min)
# pyvista_ndarray is based on numpy array!


