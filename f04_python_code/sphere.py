# https://stackoverflow.com/questions/58979478/how-to-generate-a-cone-in-3d-numpy-array

import numpy as np
import matplotlib.pyplot as plt


def sphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]

    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


x = np.arange(-10, 10)
y = np.arange(-10, 10)
z = np.arange(-10, 10)
sp = sphere((len(x), len(y), len(z)), 1, (5, 5, 5))
m = np.shape(sp)
xx, yy = np.meshgrid(x, y)

z = np.ones(np.shape(sp))

z[sp == False] = np.nan

# plot
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z, color='blue')
plt.show()
