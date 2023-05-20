import os

root_before = './before'
root_after = './after'
blur = 'blur'
grayscale = 'grayscale'
mask = 'mask'
noise = 'noise'
resize = 'resize'

list_of_folders = [blur, grayscale, mask, noise, resize]

# create base root folders if they don't exist
os.makedirs(root_before)
os.makedirs(root_after)
cycle trought the list of folder names to create    
for folder in list_of_folders:
    # create a child folder for before root path - as a string
    before_full_path = os.path.join(root_before, folder)
    # create a child folder for after root path - as a string
    after_full_path = os.path.join(root_after, folder)
    # create before full path child folder string name
    os.makedirs(before_full_path)
    # create after full path child folder string name
    os.makedirs(after_full_path)