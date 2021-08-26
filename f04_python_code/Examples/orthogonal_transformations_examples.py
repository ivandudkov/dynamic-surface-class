# For computations and plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Partly based on: https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot

vectors_start = np.array([[0, 0, 0],  # unit X arrow coordinate
                          [0, 0, 0],  # unit Y arrow coordinate
                          [0, 0, 0],
                          [1, 1, 1]])  # unit Z arrow coordinate

# Assigning the vectors itself. Let them be orthogonal.
vectors_itself = np.array([[1, 0, 0],  # unit X vector direction
                           [0, 1, 0],  # unit Y vector direction
                           [0, 0, 1],
                           [0.5, 0.9, 3]])  # unit Z vector direction


# Orthogonal transformations
# Rotation


# Shift

# Convert Matrix's columns into row vectors (2D array to 1D arrays)
i0 = vectors_start[:, 0]  # x
j0 = vectors_start[:, 1]  # y
k0 = vectors_start[:, 2]  # z

i1 = vectors_itself[:, 0]
j1 = vectors_itself[:, 1]
k1 = vectors_itself[:, 2]

# 3D plot
ax = plt.figure().add_subplot(projection='3d')
ax.quiver(i0, j0, k0, i1, j1, k1, normalize=False)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

plt.show()
