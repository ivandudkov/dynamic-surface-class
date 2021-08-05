# Define a function that searches for files with defined extension
# and returns a list of full paths for these files
import os

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
                print("%d. %s" % (count, filename))
                count += 1
    print("\n")
    return paths

# # Test
# test_paths = find_file_with_extension('.m')
# print(test_paths)
