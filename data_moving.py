import os
import shutil

# Function to read the list of filenames from a text file
def read_files_to_keep(file_path):
    with open(file_path, 'r') as file:
        files = file.read().splitlines()
    return files

# Path to the text file containing the list of filenames to keep
file_list_path = '/home/abdallah/Deepfake-Detection/dataset/Metadata/test_list/1_fake/LDM_fake_test_list.txt'  # Update with the path to your text file

# Directory containing the files
source_dir = '/home/abdallah/Deepfake-Detection/dataset/LDM'  # Update with the path to your source directory

# Directory to move the other files to
destination_dir = '../../scratch/abdallah/Dataset/LDM'  # Update with the path to your destination directory

# Read the list of filenames to keep
files_to_keep = read_files_to_keep(file_list_path)

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate through the files in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    # Check if the file is not in the list of files to keep
    if filename not in files_to_keep:
        # Move the file to the destination directory
        shutil.move(file_path, os.path.join(destination_dir, filename))

print("Files moved successfully.")
