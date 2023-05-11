import os

# gather all the files under the subfolders data/fake/subfolder/ into data/fake_flattened/

# get the current working directory
cwd = os.getcwd()

# get the path to the data folder
data_path = os.path.join(cwd, 'data/fake/')

# get the path to the flattened folder
flattened_path = os.path.join(cwd, 'data/fake_flattened/')

# get the list of subfolders in the data folder
subfolders = os.listdir(data_path)

# loop through the subfolders
for subfolder in subfolders:
    # get the path to the subfolder
    subfolder_path = os.path.join(data_path, subfolder)
    # get the list of files in the subfolder
    files = os.listdir(subfolder_path)
    # loop through the files
    for file in files:
        # get the path to the file
        file_path = os.path.join(subfolder_path, file)
        # get the path to the flattened file
        flattened_file_path = os.path.join(flattened_path, file)
        # copy the file to the flattened folder
        os.system('cp ' + file_path + ' ' + flattened_file_path)

