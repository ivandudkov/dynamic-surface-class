from find_files_with_extension import find_file_with_extension
from import_xyz import read_xyz_file_r2
from f04_python_code.reg_grid_3d import RegGrid3D

import os
import numpy as np

# Testing RegGrid "create", "add" and "plot" functions

# Turn off scientific notation
# np.set_printoptions(suppress=True)
#
# if 'xyz_array' in locals():
#     print("kek")
#     pass
# else:
#     xyz_paths = find_file_with_extension('.xyz', path=os.path.abspath('..'))
#     xyz_array = read_xyz_file_r2(xyz_paths[2], separator=',', header_length=1)

b = RegGrid3D(1, 1)

x
y
z

b.area_of_influence()

#b.create(xyz_array[0, :], xyz_array[1, :], xyz_array[2, :])

#b.add(xyz_array[0, 100:10000], xyz_array[1, 100:10000], xyz_array[2, 100:10000])
# b.filter_sd(3)

# b.plot()



