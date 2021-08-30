# For computations and plotting
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin


# Orthogonal transformations
# Theory: https://mathworld.wolfram.com/EulerAngles.html
# According to Euler's rotation theorem, any rotation may be described using three angles.
# If the rotations are written in terms of rotation matrices D, C, and B, then a general
# rotation A can be written as: A = B + C + D
#
# Euler Angels:
# The three angles giving the three rotation matrices are called Euler angles.
# There are several conventions for Euler angles, depending on the axes about which
# the rotations are carried out. Write the matrix A as:
# A=[[a_(11) a_(12) a_(13)],
#    [a_(21) a_(22) a_(23)],
#    [a_(31) a_(32) a_(33)]].
#
# The so-called "x-convention," illustrated above, is the most common definition.
# In this convention, the rotation given by Euler angles (phi,theta,psi), where:
# 1. the first rotation is by an angle phi about the z-axis using D,
# 2. the second rotation is by an angle theta in [0,pi] about the former x-axis (now x^') using C, and
# 3. the third rotation is by an angle psi about the former z-axis (now z^') using B.
#
# But there are many conventions for the angles are in common use.
#
# 1. Convention from my lectures:
# alpha - precession angle
# D = [[cos(alpha), -sin(alpha), 0],
#      [sin(alpha),  cos(alpha), 0],
#      [    0     ,      0     , 1]]
# beta - nutation angle
# C = [[1,  0     ,      0      ],
#      [0, cos(beta), -sin(beta)],
#      [0, sin(beta),  cos(beta)]]
# gamma - self-rotation angle
# B = [[cos(gamma), -sin(gamma), 0],
#      [sin(gamma),  cos(gamma), 0],
#      [    0     ,      0     , 1]]
#
# 2. In the "xyz (pitch-roll-yaw) convention," theta is pitch, psi is roll, and phi is yaw.
#
# D	=	[[ cos(phi), sin(phi), 0],
#        [-sin(phi), cos(phi), 0],
#        [    0    ,    0    , 1]]
#
# C	=	[[cos(theta), 0, -sin(theta)],
#        [     0    , 1,      0     ],
#        [sin(theta), 0,  cos(theta)]]
#
# B	=	[[1,          0,     0   ],
#         [0,  cos(psi), sin(psi)],
#         [0, -sin(psi), cos(psi)]]
#
# 3. Convention that we used in Lab A: Roll-Pitch-Yaw convention
# Remember! Matrix multiplication order for rotation: 𝑅 = 𝑅(𝑌) ⋅ 𝑅(𝑃) ⋅ 𝑅(𝑅)
# Rotation around x-axis. Roll. R(R)
# Rx = np.array([[1, 0, 0],
#                [0, cos(psi), -sin(psi)],
#                [0, sin(psi), cos(psi)]])
# Rotation around y-axis. Pitch. R(P)
# Ry = np.array([[cos(theta), 0, sin(theta)],
#                [0, 1, 0],
#                [-sin(theta), 0, cos(theta)]])
# Rotation around z-axis. Yaw R(Y)
# Rz = np.array([[cos(phi), -sin(phi), 0],
#                [sin(phi), cos(phi), 0],
#                [0, 0, 1]])
#

# Define function for rotation
def roll_pitch_yaw_rotation(xyz_vector, angles_vector):
    # Rotation around x-axis. Roll. R(R)
    psi = angles_vector[0]
    theta = angles_vector[1]
    phi = angles_vector[2]
    Rx = np.array([[1, 0, 0],
                   [0, cos(psi), -sin(psi)],
                   [0, sin(psi), cos(psi)]])

    # Rotation around y-axis. Pitch. R(P)
    Ry = np.array([[cos(theta), 0, sin(theta)],
                   [0, 1, 0],
                   [-sin(theta), 0, cos(theta)]])

    # Rotation around z-axis. Yaw R(Y)
    Rz = np.array([[cos(phi), -sin(phi), 0],
                   [sin(phi), cos(phi), 0],
                   [0, 0, 1]])

    # 𝑅 = 𝑅z(𝑌) ⋅ 𝑅y(𝑃) ⋅ 𝑅x(𝑅)
    rotated_matrix = Rz @ Ry @ Rx @ xyz_vector
    return rotated_matrix


# Shift
def shift_coordinates(a):
    a = a
    pass


vectors_start = np.array([[0, 0, 0],  # unit X arrow coordinate
                          [0, 0, 0],  # unit Y arrow coordinate
                          [0, 0, 0],
                          [1, 1, 1]])  # unit Z arrow coordinate

# Assigning the vectors itself. Let them be orthogonal.
vectors_itself = np.array([[1, 0, 0],  # unit X vector direction
                           [0, 1, 0],  # unit Y vector direction
                           [0, 0, 1],
                           [0.5, 0.9, 3]])  # unit Z vector direction

vectors_rotated = np.array([[1, 0, 0],  # unit X vector direction
                           [0, 1, 0],  # unit Y vector direction
                           [0, 0, 1],
                           [0.5, 0.9, 3]])  # unit Z vector direction

for vector, i in zip(vectors_itself[:, 0:None], range(vectors_itself.shape[0])):
    vectors_rotated[i, :] = roll_pitch_yaw_rotation(vector, [np.pi/4, 0, 0])

# Convert Matrix's columns into row vectors (2D array to 1D arrays)
i0 = vectors_start[:, 0]  # x
j0 = vectors_start[:, 1]  # y
k0 = vectors_start[:, 2]  # z

i1 = vectors_itself[:, 0]
j1 = vectors_itself[:, 1]
k1 = vectors_itself[:, 2]

i2 = vectors_rotated[:, 0]
j2 = vectors_rotated[:, 1]
k2 = vectors_rotated[:, 2]

# 3D plot
ax = plt.figure().add_subplot(projection='3d')
ax.quiver(i0, j0, k0, i1, j1, k1, normalize=False)
ax.quiver(i0, j0, k0, i2, j2, k2, color='r', normalize=False)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

plt.show()
