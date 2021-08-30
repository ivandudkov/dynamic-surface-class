# PyVista tutorial
# https://www.youtube.com/watch?v=FmNmRBsEBHE&ab_channel=SoftwareUnderground
# https://github.com/banesullivan/transform-2021

import numpy as np
import pyvista as pv

# 1. Let's create a 3D array of XYZ points:
points = np.linspace(0, 10, 120).reshape(40, 3)

# 2. Next we create PolyData object
point_cloud = pv.PolyData(points)

# Print the information of point_cloud object
print(point_cloud)

# We also can do sanity check, to be sure that mesh-points and our points are
# actually the same:
print(np.allclose(points, point_cloud.points))  # result == True

# 3. That's it, now we can plot our point cloud
point_cloud.plot(eye_dome_lighting=True)  # eye_dome_lighting - good option, makes points more distinguishable.

# 4. Okay, but how about point's color? To colorize the points we should create
# an attribute for our meshgrid. Let's do this:
point_cloud['z_values'] = points[:, 2]

# 4.1 Let's plot it again, now points will be colorized. Also, let's visualize points
# as spheres
point_cloud.plot(render_points_as_spheres=True)

# 5. Next, we will add vectors to our points - i.e. data array with more than one scalar value
# 5.1 Create new, random point cloud
points = np.random.rand(100, 3) * 10

# 5.2 Make PolyData object
point_cloud = pv.PolyData(points)


# 6. Let's define some function that will add vectors for every
# node in point cloud:

def compute_vectors(mesh):
    origin = mesh.center
    vectors = mesh.points - origin
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]  # this will create column vector
    return vectors


# 6.1 Apply that function
vectors = compute_vectors(point_cloud)

# 7. Add vectors as attribute array
point_cloud['vectors'] = vectors

# 8. Now we can make arrows using those vectors using glyph filter
# see glyph_example for more details
arrows = point_cloud.glyph(orient='vectors', scale=False, factor=1)

# 9. Display the arrows
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, color='maroon', point_size=10., render_points_as_spheres=True)
plotter.add_mesh(arrows, color='lightblue')

plotter.show_grid()
plotter.show()



