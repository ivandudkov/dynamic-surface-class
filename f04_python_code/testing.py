from find_files_with_extension import find_file_with_extension
from import_xyz import read_xyz_file_r2
from f04_python_code.reg_grid_3d import RegGrid3D

import os
import numpy as np

# Turn off scientific notation
np.set_printoptions(suppress=True)

xyz_paths = find_file_with_extension('.xyz', path=os.path.abspath('..'))
xyz_array = read_xyz_file_r2(xyz_paths[2], separator=',',header_length=1)

b = RegGrid3D(0.5,1)
b.Create(xyz_array[0,0:1000], xyz_array[1,0:1000], xyz_array[2,0:1000])
b.FilterSD(5)
print("Kekekke")

b.Plot()
