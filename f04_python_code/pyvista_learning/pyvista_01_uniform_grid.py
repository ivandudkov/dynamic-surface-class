# PyVista tutorial
# https://www.youtube.com/watch?v=FmNmRBsEBHE&ab_channel=SoftwareUnderground
# https://github.com/banesullivan/transform-2021

import numpy as np
import pyvista as pv

# 1. Let's create 3D numpy array
# To do that, firstly we create 1D array of values
values = np.linspace(0, 10, 1000)

# And after convert that array using .reshape method into
# 3D numpy array with dims: x = 20, y = 5, z = 10 i.e. 20x5x10 array
values = values.reshape((20, 5, 10))

# 2. Next, we create the spatial reference for our data - PyVista mesh object
grid = pv.UniformGrid()  # Just an instance of the uniform grid object

# 3. After we should populate our object. But before, we have to set up
# dimensionality of our mesh grid. It will slightly differ depending on how we want
# populate our mesh. Do we want populate grid points or grid cells?

# 3.1 Populating grid (mesh) cells:
grid.dimensions = np.array(values.shape) + 1  # where
# np.array(values.shape) is an 1D array containing dimensions of our values
# and + 1 - because we want to inject our values on the CELL data

# 3.2 Populating grid (mesh) points:
# grid.dimensions = np.array(values.shape)
# in that case we are not adding one, because we want to inject our values
# on the POINT data (i.e. number of points == number of values)

# 4. Edit the spatial reference: grid origin and spacing
grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
grid.spacing = [1, 5, 2]  # These are cell (voxel) sizes along each axis

# 5. Add the data values to the cell data
grid.cell_arrays['values'] = values.flatten(order='F')  # to do that correctly
# we should flatten our array. F-order is a specific of the pyvista array, if we
# do flatten for pyvista we should keep that order 'F' (I really don't know why)

# 5.1 Visualize the mesh:
grid.plot(show_edges=True)

# 6. Add the data values to the point data
# grid.point_arrays['values'] = values.flatten(order='F')  # The same as for cell array
# we should flatten our 3D numpy array

# 6.1 Visualize the mesh:
# grid.plot(show_edges=True)



