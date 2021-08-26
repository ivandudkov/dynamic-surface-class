import numpy as np  # for numerical computations and array handling
import matplotlib.pyplot as plt  # for plotting
import matplotlib.cm as cm  # for colormap handling

# Partly based on: https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot
# And: https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html

vectors_start = np.array([[0, 0, 0],  # unit X arrow coordinate
                          [0, 0, 0],  # unit Y arrow coordinate
                          [0, 0, 0],
                          [1, 1, 1]])  # unit Z arrow coordinate

# Assigning the vectors itself. Let them be orthogonal.
vectors_itself = np.array([[1, 0, 0],  # unit X vector direction
                           [0, 1, 0],  # unit Y vector direction
                           [0, 0, 1],
                           [0.5, 0.9, 3]])  # unit Z vector direction

# Convert Matrix's columns into row vectors (2D array to 1D arrays)
i0 = vectors_start[:, 0]  # x
j0 = vectors_start[:, 1]  # y
k0 = vectors_start[:, 2]  # z

i1 = vectors_itself[:, 0]
j1 = vectors_itself[:, 1]
k1 = vectors_itself[:, 2]

# Calculate colors 1D array. Let's define it as vector magnitudes in coordinate space units
colors = np.zeros(vectors_itself.shape[0])
for vector, i in zip(vectors_itself[:, 0:None], range(vectors_itself.shape[0])):
    colors[i] = np.linalg.norm(vector)

# Case 1. Colors for 2D quiver plot:
colors_2D = colors
# Normalize our colors to match
# it colormap domain which is [0, 1]
colors_2D = colors_2D / (colors_2D.ptp() + colors_2D.min())

# Pick colormap and define color.
# it has domain [0, 1] and that's
# why it is so important to keep
# color array elements in that range
# (i.e. it is why normalization is needed)
colors_2D = cm.viridis(colors_2D)

# Case 2. Colors for 3D quiver plot:
colors_3D = colors

# Flatten array and normalize
colors_3D = (colors_3D.ravel()) / (colors_3D.ptp() + colors_3D.min())

# Repeat for each body line and two head lines. Yeah, the fact that we also should
# colorize arrow's head is just weird
colors_3D = np.concatenate((colors_3D, np.repeat(colors_3D, 2)))

# Colormap for our colours. Remember, it has [0, 1] domain.
# That's why we did previous two steps
colors_3D = plt.cm.viridis(colors_3D)

# 3D plot
ax = plt.figure().add_subplot(projection='3d')
ax.quiver(i0, j0, k0, i1, j1, k1, color=colors_3D, normalize=False)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

plt.show()

# 2D Plot
ax = plt.figure().add_subplot()
ax.quiver(i0, j0, i1, j1, color=colors_2D)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])

plt.show()


# From https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot
import matplotlib.pyplot as plt
import numpy as np

# Make the grid
x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

# Make the direction data for the arrows
u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))

# Color by azimuthal angle
c = np.arctan2(v, u)
# Flatten and normalize
c = (c.ravel() - c.min()) / c.ptp()
# Repeat for each body line and two head lines
c = np.concatenate((c, np.repeat(c, 2)))
# Colormap
c = plt.cm.hsv(c)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(x, y, z, u, v, w, colors=c, length=0.1, normalize=True)
plt.show()
