# PyVista tutorial
# https://www.youtube.com/watch?v=FmNmRBsEBHE&ab_channel=SoftwareUnderground
# https://github.com/banesullivan/transform-2021

import numpy as np
import pyvista as pv
from pyvista import examples

# 1. Create a simple meshgrid using numpy
x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
x, y = np.meshgrid(x, y)

# Define Z-values by parametric equation
# It will be a regular sinc function
r = np.sqrt(x**2 + y**2)
z = np.sin(r)

# 2. Pass the numpy meshgrid to PyVista
# to create a pv.StructuredGrid() object
grid = pv.StructuredGrid(x, y, z)  # Pass the x, y, z values directly to the StructuredGrid
grid.plot()

# 2.1 We also can plot a mean curvature of the surface
grid.plot_curvature(clim=[-1, 1])

# 2.2 Also, points of that grid can be accessed by .points
print(grid.points)  # PyVista array - just a Numpy array

# 3. StructuredGrid from XYZ points
# Quite often, you might be given a set of coordinates (XYZ points) in a simple tabular format
# where there exists some structure such that grid could be built between the nodes you have.
# A great example is found in pyvista-support#16 where a structured grid that is rotated from
# the cartesian reference frame is given as just XYZ points. In these cases, all that is needed
# to recover the grid is the dimensions of the grid (nx by ny by nz) and that the coordinates
# are ordered appropriately.
#
# For this example, we will create a small dataset and rotate the coordinates
# such that they are not on orthogonal to cartesian reference frame.
def make_point_set():
    """Ignore the contents of this function. Just know that it returns an
    n by 3 numpy array of structured coordinates."""
    n, m = 29, 32
    x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=m) + np.random.uniform(-5, 5, size=m)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    foo = pv.PolyData(points)
    foo.rotate_z(36.6)
    return foo.points

# Get the points as a 2D NumPy array (N by 3)
points = make_point_set()


# Now pretend that the (n by 3) NumPy array above are coordinates that you have,
# possibly from a file with three columns of XYZ points.
#
# We simply need to recover the dimensions of the grid that these points make
# and then we can generate a :class:pyvista.StructuredGrid mesh.
#
# Let's preview the points to see what we are dealing with:
# Цe could connect the points as a structured grid. All we
# need to know are the dimensions of the grid present. In this
# case, we know (because we made this dataset) the dimensions
# are [29, 32, 1], but you might not know the dimensions of
# your pointset. There are a few ways to figure out the
# dimensionality of structured grid including:

# 1) Manually counting the nodes along the edges of the
# pointset
# 2) Using a technique like principle component analysis
# to strip the rotation from the dataset and count
# the unique values along each axis for the new;y
# projected dataset.


# Once you've figured out your grid's dimensions,
# simple create the :class:`pyvista.StructuredGrid`
# as follows:
mesh = pv.StructuredGrid()
# Set the coordinates from the numpy array
mesh.points = points
# set the dimensions
mesh.dimensions = [29, 32, 1]

# and then inspect it!
mesh.plot(show_edges=True, show_grid=True, cpos="xy")

# 4. Extending a 2D StructuredGrid to 3D
# A 2D class pyvista.StructuredGrid mesh can be extended into
# a 3D mesh. This is highly applicable when wanting to create
# a terrain following mesh in earth science research applications.

# For example, we could have a :class:pyvista.StructuredGrid of a
# topography surface and extend that surface to a few different
# levels and connect each "level" to create the 3D terrain
# following mesh (maybe that way we can take into account
# volumetric backscattering????"

# A simple example by extending the wave mesh to 3D:
# 4.1 Load example
struct = examples.load_structured()
struct.plot(show_edges=True)

# Copy the top layer (i.e. 2D surface)
top = struct.points.copy()

# Extend bottom to a few layers deep
bottom = struct.points.copy()
bottom[:,-1] = -10.0 # Wherever you want the plane

# Create StructuredGrid object and populate it
vol = pv.StructuredGrid()
vol.points = np.vstack((top, bottom))
vol.dimensions = [*struct.dimensions[0:2], 2]
vol.plot(show_edges=True)










