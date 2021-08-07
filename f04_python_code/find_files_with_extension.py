import os

# Define a function that searches for files with defined extension
# and returns a list of full paths for these files

# If function doesn't find any file in the parent directory,
# it will search for one directory up

# August 2021
# Author: Ivan Dudkov
# Affiliations:
# P.P Shirshov's Institute of Oceanography, Atlantic Branch, Kaliningrad, Russia. (2018 - ????)
# Center for Coastal and Ocean Mapping, University of New Hampshire. Durham, USA. (2020-2021)

def find_file_with_extension(extension, path=os.getcwd()):
    # Check the top path existence
    if os.path.exists(path):
        print(('Searching *%s files in directory:' + path + '\n') % extension)
    else:  # Raise a meaningful error
        raise RuntimeError('Path either not exists or not correct ' + path)

    paths = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            fextension = os.path.splitext(filename)[1]
            if fextension == extension:
                paths.append(os.path.abspath(os.path.join(dirpath, filename)))
                print("Files found:\n%d. %s" % (count, filename))
                count += 1
    count = 0
    if len(paths) == 0:
        os.chdir(os.path.dirname(path))
        path_up = os.getcwd()
        print("Searching was unsuccessful. Trying to search in one directory up:\n%s" % path_up)

        for dirpath, dirnames, filenames in os.walk(path_up):
            for filename in filenames:
                fextension = os.path.splitext(filename)[1]
                if fextension == extension:
                    paths.append(os.path.abspath(os.path.join(dirpath, filename)))
                    print("Files found:\n%d. %s" % (count, filename))
                    count += 1

    print("\n")
    return paths

# Test
# test_paths = find_file_with_extension('.txt')
# print(test_paths)
