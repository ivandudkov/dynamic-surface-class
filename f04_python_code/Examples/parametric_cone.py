# Implementation of the cone through parametric representation
# using vectors array

# Also, ray tracing and refraction - remember that!

# Convention - everything relative to sonar's reference frame
# x - positive forward
# y - positive starboard
# z - positive down
# Maybe we should assume z as a time?

# import libraries
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# firing angle?
# beam angle - along- and across-track
# focus on simple cone shape
# intersect the vectors with the surface

# 1. Create a surface. There are 2 surfaces:
# a) Flat surface with constant depth;
# b) Rough seafloor with variable depth.

# 100x100 grid
X, Y = np.meshgrid(np.arange(-50, 51, 1), np.arange(-50, 51, 1))

# depth values = 20m
Z_flat = np.ones(X.shape) * 20.

# random depth values
Z_rough = np.sort(np.ones(X.shape) * 20 * np.random.rand(X.shape[0], X.shape[1]), kind='heapsort')

# Now specify the parametric cone
# General equation: x^2/a^2 + y^2/b^2 = z^2/c^2
# Parametric equation:
#             x = ((h - u)/h)*r*cos(theta)
#             y = ((h - u)/h)*r*sin(theta)
#             z = u
#             where u is [0, h]; theta is [0, 2pi]
#             r - base radius; h - height

n = 50  # number of vectors
h = 25  # cone height
op_angle = np.arctan(1)  # cone opening angle = 45 degrees (in radians)
# Using formula phi = 2*arctan(r/h) -> tan(phi)/2 = r/h -> r = tan(phi)*h/2
# We derive cone base radius
r = np.tan(op_angle) * h / 2
# Resolution in depth. Should be equal grid resolution
res = 1
# Define the theta and u parameters. Remember z = u
theta = np.arange(0, 2 * np.pi + 0.1, 0.1)
u = np.arange(0, h + res, res)

# Create a meshgrid for u and theta:
theta_cone, u_cone = np.meshgrid(theta, u)
# Calculate cone vectors
x_cone = ((h - u_cone) / h) * r * np.cos(theta)
y_cone = ((h - u_cone) / h) * r * np.sin(theta)
z_cone = u_cone

# Plot the grid
# plt.contourf(X, Y, Z_rough, cmap='jet_r')
# plt.colorbar()
# plt.show()

# plot the plane
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(x_cone, y_cone, z_cone, color='blue')
# plt3d.plot_surface(X, Y, Z_flat, color='blue')
# #plt3d.contour3D(xx, yy, cone_zz, 0, cmap='binary')
# plt.show()

# PyVista stuff:

# Cone data:
# 1. Create a 3D cone points array
points = np.column_stack((x_cone.ravel(order='F'),
                          y_cone.ravel(order='F'),
                          z_cone.ravel(order='F')))
# 2. Assign values to PolyData object by
# passing in there flatten arrays
cone_vc = pv.PolyData(points)

# 3. Create 2D surface mesh
xyz_point_cloud = np.column_stack((X.ravel(order='F'),
                                   Y.ravel(order='F'),
                                   Z_rough.ravel(order='F')))

# Add Z-values attribute
depth_points = pv.PolyData(xyz_point_cloud)
# Create triangle mesh surface
seabed = depth_points.delaunay_2d(alpha=5)


# 4. Perform ray trace
# Source: # https://github.com/pyvista/pyvista-support/issues/173
def ray_trace(start, stop, mesh, conemesh):
    """Pass two same sized arrays of the start and stop coordinates for all rays"""
    origin = conemesh.center
    assert start.shape == stop.shape
    assert start.ndim == 2
    # Launch this for loop in parallel if needed
    # start = start + origin
    # stop = stop + origin
    zeroth_cellids = []
    count = 0
    intersections = np.zeros((1, 3))
    for i in range(len(start)):
        q, ids = mesh.ray_trace(start[i], stop[i])
        if len(ids) < 1:
            v = None
        else:
            v = ids[0]
            if count == 0:
                intersections = q
                count += 1
            else:
                intersections = np.append(intersections, q)
                count += 1
        zeroth_cellids.append(v)
    return np.array(zeroth_cellids), intersections.reshape(count, 3)


# for i in range(len(x_cone[0, :])):
#     for j in range(len(x_cone[:, i])):
#
start = np.array([[x_cone[0, 0], y_cone[0, 0], z_cone[0, 0]], [x_cone[0, 20], y_cone[0, 20], z_cone[0, 20]]])
stop = np.array([[x_cone[-1, 0], y_cone[-1, 0], z_cone[-1, 0]], [x_cone[-1, 20], y_cone[-1, 20], z_cone[-1, 20]]])
cell_ids, intersections = ray_trace(start, stop, seabed, cone_vc)

# Extract affected cells
cells = seabed.extract_cells(cell_ids)

# Create mesh for intersections:
intersect = pv.PolyData(intersections)


def compute_vectors(mesh):
    origin = [0, 0, 20]
    vectors = mesh.points - origin
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]  # this will create column vector
    return vectors

# 6.1 Apply that function
vectors = compute_vectors(intersect)

# 7. Add vectors as attribute array
intersect['vectors'] = vectors
arrows = intersect.glyph(orient='vectors', scale=False, factor=1)


# 3. Create scene:
plotter = pv.Plotter()
plotter.add_mesh(cone_vc)
plotter.add_mesh(intersections, render_points_as_spheres=True, point_size=15, color='red')
plotter.add_mesh(arrows, color='lightblue', opacity=0.5)
for i in range(len(start[:, 0])):
    ray = pv.Line(start[i, :], stop[i, :])
    plotter.add_mesh(ray, color="blue", line_width=5, label="Ray Segment")
plotter.add_mesh(seabed)

plotter.show_grid()
plotter.show()
