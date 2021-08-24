import numpy as np
import matplotlib.pyplot as plt

# This code is about the plane definition (and visualisation)
# in python code using plane equation

# This script partly based on analytical geometry course:
# https://youtu.be/qft7hXY8Cw0 (in russian. Пенской А. В. - Аналитическая геометрия;
# Penskoy A. V. - Analytical Geometry. Lomonosov Moscow State University, faculty of mathematics and mechanics)

# and partly on: https://stackoverflow.com/questions/19410733/how-to-draw-planes-from-a-set-of-linear-equations-in-python

# General equation for a plane is:
# Ax + By + Cz + D = 0
# The normal for a plane is a vector (A, B, C)

# To build a plane in python we need to convert x,y,z
# grid into plane's values (I don't know how I can call it)
# according to the plane equation that we picked

# To perform it, we can use next knowledge:
# It is known, that if A != 0 and B, C == 0, then
# dot A has coordinates (-D/A, 0, 0)
# i.e. if we know, that B and C == 0, then
# Ax + 0y + 0z + D = 0 -> Ax + D = 0 -> Ax = -D -> x = -D/A
# for B = (0, D/B, 0)
# for C = (0, 0, -D/C)

# Then, using that knowledge we can define those points that satisfy
# with equation of our plane

# Also, if A,B,C are known, we are able to find D
# D = dot product of vectors v = (-D/A, D/B, -D/C) - dot in a plane;
# and c = (A, B, C) - plane's normal
# D = v * c or (v, c) or np.dot(v, c)

# Let's take plane's equation:
# -4x1 + 5x2 + 9x3 = -9 <=> -4x1 + 5x2 + 9x3 + 9 = 0 (general view)
# A = -4; B = 5; C = 9; D = 9

# 2. Define A, B, C and D:
A = -4.
B = 5.
C = 9.
D = 9.

# 1. Create an array of a normal vector:
normal = np.array([A, B, C])
# 2. Define a point that lie on the plane
point = np.array([-D / A, D / B, -D / C])
# 3. Create values for x and y:
xx = np.linspace(-30, 30, 301)  # array contains x coordinates, row-vector
yy = np.linspace(-30, 30, 301)[:, None]  # array contains y coordinates, column vector

# 4. Calculate corresponding z
# According to the general equation, z equals:
# Ax + By + Cz + D = 0 -> Cz = -Ax - By - D
# -> z = (-Ax - By - D)/C or -(Ax + By + D)/C
zz = -1 * (A * xx + B * yy + D) * 1. / C

# plot the plane
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, zz, color='blue')
# Plot the point on that plane
plt3d.scatter(point[0], point[1], point[2])
# Plot the normal vector from that point
plt3d.plot([point[0], normal[0]], [point[1], normal[1]], [point[2], normal[2]])
# Plot the orthogonal vectors to normal vector
a = np.array([-5, 5, -5])
b = np.array([5, -5, 5])
dot_prod_b = np.dot(normal, b)
dot_prod_a = np.dot(normal, a)

plt3d.plot([point[0], a[0]], [point[1], a[1]], [point[2], a[2]])
plt3d.plot([point[0], b[0]], [point[1], b[1]], [point[2], b[2]])


plt.show()
