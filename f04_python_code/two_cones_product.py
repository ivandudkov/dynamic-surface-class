import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import norm

# General equation for a plane:
# Ax + By + Cz + D = 0

# General equation for a cone
# x^2/a^2 + y^2/b^2 = z^2/c^2
# or
# x^2/a^2 + y^2/b^2 - z^2/c^2 = 0
# z = =-sqrt(A^2*x^2 + B^2*y^2)
# negative z - low portion of a cone (we need this, if we are using negative depths)

# General equation for an ellipsis
# x^2/a^2 + y^2/b^2 = 1
# or
# x^2/a^2 + y^2/b^2 - 1 = 0
#
# If our center has an offset:
# (x-x0)^2/a^2 + (y-y0)^2/b^2 = 1

# System of equations:
# Ax + By + Cz + D = 0
# x^2/a^2 + y^2/b^2 - z^2/c^2 = 0
# x^2/a^2 + y^2/b^2 - 1 = 0

# 1. Create x, y and z coordinates
x = np.linspace(-6, 6, 25)
y = np.linspace(-6, 6, 25)
z = np.linspace(-6, 6, 25)

# 2. Create meshgrid for x and z coordinates (cartesian space)
xx, yy = np.meshgrid(x, y)
xx_shape = np.shape(xx)
yy_shape = np.shape(yy)

# 3. Declare a plane equation
# plane_z = -(Ax + By + D)/C
# y[:, 0].T - column vector y
plane_zz = np.ones(xx_shape) * np.nan
D = -5

abc_plane = np.array([0, 0, 1])
for i in range(xx_shape[0]):
    for j in range(yy_shape[0]):
        plane_zz[i, j] = -1 * (abc_plane[0] * xx[i, j] + abc_plane[1] * yy[j, j].T + D) / abc_plane[2]

# 4. Declare a cone equation
# General: x^2/a^2 + y^2/b^2 = z^2/c^2
# Parametric: x = ((h - u)/h)*r*cos(theta)
#             y = ((h - u)/h)*r*sin(theta)
#             z = u
#             where u is [0, h]; theta is [0, 2pi]
# r - base radius; h - height
# Cone Z = z = sqrt((x^2/a^2 + y^2/b^2)*c^2)

n = 800
h = 10
op_angle = np.arctan(3)
r = np.tan(op_angle)*h/2# Using formula phi = 2*arctan(r/h) -> tan(phi)/2 = r/h -> r = tan(phi)*h/2
res = 1
theta = np.arange(0, 2*np.pi, 0.01) + np.pi/4
u = np.arange(0, h+res, res)

# Create a meshgrid for u and theta:
theta_cone, u_cone = np.meshgrid(theta, u)

# x_cone = np.linspace(0, 10, n)
# y_cone = np.linspace(0, 10, n)

# Source: https://mathworld.wolfram.com/EllipticCone.html
# Elliptic cone 1
a = 2
b = 5

x_cone = a*((h - u_cone)/h)*r*np.cos(theta)
y_cone = b*((h - u_cone)/h)*r*np.sin(theta)
z_cone = u_cone

# Elliptic cone 2
a2 = 10
b2 = 1
x_cone2 = a2*((h - u_cone)/h)*r*np.cos(theta)
y_cone2 = b2*((h - u_cone)/h)*r*np.sin(theta)

# Cone - intersection of both cones
xx = x_cone - x_cone2
yy = y_cone - y_cone2

xx[-0.05<xx<0.05] = xx
yy[-0.05<xx<0.05] = yy
# plot the plane
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(x_cone, y_cone, z_cone, color='blue')
plt3d.plot_surface(x_cone2, y_cone2, z_cone, color='blue')
plt3d.plot_surface(xx, yy, z_cone, color='blue')
#plt3d.contour3D(xx, yy, cone_zz, 0, cmap='binary')
plt.show()
