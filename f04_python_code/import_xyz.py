import os.path
import numpy as np

# Define a function that read XYZ files
# and returns a numpy array

# August 2021
# Author: Ivan Dudkov
# Affiliations:
# P.P Shirshov's Institute of Oceanography, Atlantic Branch, Kaliningrad, Russia. (2018 - ????)
# Center for Coastal and Ocean Mapping, University of New Hampshire. Durham, USA. (2020-2021)

# Realization 1. Probably well optimized but it crashes EPOM kernel
def read_xyz_file_r1(xyz_filepath, start=0, end=None,
                     header_length=0, separator=' '):
    # Check the file existence
    if os.path.exists(xyz_filepath):
        print('Reading file: %s\n' % xyz_filepath)
    else:  # Raise a meaningful error
        raise RuntimeError('File is not exist or path is not correct')
    if end != None:
        xyz_array = np.genfromtxt(xyz_filepath, delimiter=separator, skip_header=header_length,
                                  missing_values=None, max_rows=(end - start))
    else:
        xyz_array = np.genfromtxt(xyz_filepath, delimiter=separator, skip_header=header_length,
                                  missing_values=None, max_rows=end)
    return xyz_array

    # Realization 2. Probably, the best realization.
    # It takes the least amount of memory and works well, but doesn't have some important features
    # It doesn't crashing EPOM kernel


def read_xyz_file_r2(xyz_filepath, header_length=0, separator=' '):
    # Check the file existence
    if os.path.exists(xyz_filepath):
        print('Reading file: %s\n' % xyz_filepath)
    else:  # Raise a meaningful error
        raise RuntimeError('File is not exist or path is not correct')

    header = list()
    x_list = list()
    y_list = list()
    z_list = list()

    count = 0

    with open(xyz_filepath, 'r') as xyz_file:
        for index, line in enumerate(xyz_file):
            if index in range(header_length):
                x, y, z = line.split(separator)
                header.append(x)
                header.append(y)
                header.append(z)

                count += 1
            else:
                x, y, z = line.split(separator)

                x_list.append(round(float(x), 3))
                y_list.append(round(float(y), 3))
                z_list.append(round(float(z), 3))
                count += 1
    print("Number of Rows: %d" % count)
    xyz_array = np.array([x_list, y_list, z_list])
    return xyz_array

    # Realization 3. The easiest and the most expensive one


def read_xyz_file_r3(xyz_filepath, start=0, end=None,
                     header_length=0, separator=' '):
    # Check the file existence
    if os.path.exists(xyz_filepath):
        print('Reading file: %s\n' % xyz_filepath)
    else:  # Raise a meaningful error
        raise RuntimeError('File is not exist or path is not correct')

    # Open, read and close the file
    xyz_file = open(xyz_filepath, 'r')
    xyz_file_content = xyz_file.read()
    xyz_file.close()

    xyz_lines = xyz_file_content.splitlines()
    count = 1

    header = list()
    x = list()
    y = list()
    z = list()

    # Header reading
    for header_line in xyz_lines[start: start + header_length]:
        header.append(header_line.split(separator))
        count += 1
    # Data reading
    for xyz_line in xyz_lines[start + header_length: end]:
        xyz = xyz_line.split(separator)
        x.append(round(float(xyz[0]), 3))
        y.append(round(float(xyz[1]), 3))
        z.append(round(float(xyz[2]), 3))
        count += 1
    print("%d rows were read" % count)
    xyz_array = np.array([x, y, z])

    return header, xyz_array
