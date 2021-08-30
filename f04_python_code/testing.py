from find_files_with_extension import find_file_with_extension
from import_xyz import read_xyz_file_r2
from f04_python_code.reg_grid_3d import RegGrid3D

import os
import numpy as np
import pyvista as pv

# Testing RegGrid "create", "add" and "plot" functions

# Turn off scientific notation
np.set_printoptions(suppress=True)

if 'xyz_array' in locals():
    print("kek")
    pass
else:
    xyz_paths = find_file_with_extension('.xyz', path=os.path.abspath('..'))
    xyz_array = read_xyz_file_r2(xyz_paths[2], separator=',', header_length=1)


data = xyz_array.T
mesh = pv.PolyData(data)
elev = data[:, -1]
mesh["elevation"] = elev

mesh.plot(point_size=10, eye_dome_lighting=True)

surf = mesh.delaunay_2d(alpha=1.0)
surf.plot(show_edges=False)

# b = RegGrid3D(1, 1)
#
# x = [1, 2, 3]
# y = [1, 2, 3]
# bh = [90*np.pi/180, 86*np.pi/180, 94*np.pi/180]
# h = [145*np.pi/180, 147*np.pi/180, 149*np.pi/180]
#
# b.area_of_influence(x, y, bh, h)

# b.create(xyz_array[0, :], xyz_array[1, :], xyz_array[2, :])

#b.add(xyz_array[0, 100:10000], xyz_array[1, 100:10000], xyz_array[2, 100:10000])
# b.filter_sd(3)

# b.plot()