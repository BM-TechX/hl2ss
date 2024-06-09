import json
from sys import argv  
import numpy as np  
import os  
import shutil  
from scipy.spatial.distance import pdist, squareform  
  
def load_transforms(file_path):  
    with open(file_path, 'r') as f:  
        data = json.load(f)  
    return data  
  
def frobenius_distance(matrices):  
    # Calculate pairwise Frobenius norm differences between matrices  
    flattened_matrices = [np.array(matrix).flatten() for matrix in matrices]  
    return squareform(pdist(flattened_matrices, metric='euclidean'))  
  
def filter_frames(transforms, threshold_percent):  
    matrices = [frame['transform_matrix'] for frame in transforms['frames']]  
    distances = frobenius_distance(matrices)  
    np.fill_diagonal(distances, np.inf)  # Ignore self-comparison  
  
    num_to_remove = int(len(transforms['frames']) * threshold_percent / 100)  
    removed_indices = set()  
  
    while len(removed_indices) < num_to_remove:  
        # Find the pair with the smallest distance  
        min_idx = np.argmin(distances)  
        i, j = np.unravel_index(min_idx, distances.shape)  
  
        # Add one of the frames to the removal set  
        if len(removed_indices) < num_to_remove:  
            removed_indices.add(i)  
        if len(removed_indices) < num_to_remove:  
            removed_indices.add(j)  
  
        # Set distances to infinity to avoid selecting again  
        distances[i, :] = np.inf  
        distances[:, i] = np.inf  
        distances[j, :] = np.inf  
        distances[:, j] = np.inf  
  
    # Filter out the frames  
    new_frames = [frame for idx, frame in enumerate(transforms['frames']) if idx not in removed_indices]  
    return new_frames  
  
def create_filtered_dataset(source_dir, dest_dir, threshold_percent):  
    if not os.path.exists(dest_dir):  
        os.makedirs(dest_dir)  
  
    transforms_path = os.path.join(source_dir, 'transforms.json')  
    transforms = load_transforms(transforms_path)  
  
    # Filter frames  
    filtered_frames = filter_frames(transforms, threshold_percent)  
    transforms['frames'] = filtered_frames  
  
    # Save the new transforms.json  
    with open(os.path.join(dest_dir, 'transforms.json'), 'w') as f:  
        json.dump(transforms, f, indent=4)  
  
    # Copy the necessary rgb and poses files  
    for folder in ['rgb', 'poses']:  
        source_folder = os.path.join(source_dir, folder)  
        dest_folder = os.path.join(dest_dir, folder)  
        os.makedirs(dest_folder, exist_ok=True)  
  
        for frame in filtered_frames:  
            file_name = os.path.basename(frame['file_path'])
            #TXT source file path in poses
            if folder == 'poses':
                file_name = file_name.replace('.png', '.txt')  
            src_file_path = os.path.join(source_folder, file_name)  
            dst_file_path = os.path.join(dest_folder, file_name)  
            shutil.copy(src_file_path, dst_file_path)  
  
# Configuration  
source_directory = argv[1]  
destination_directory = argv[2]  
threshold_percent = int(argv[3])  # Percentage of images to filter out  le
  
create_filtered_dataset(source_directory, destination_directory, threshold_percent)  
