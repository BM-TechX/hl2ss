import numpy as np  
import os  
import random  
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
  
def read_pose_matrix(filepath):  
    """ Read a pose matrix from a text file. """  
    with open(filepath, 'r') as file:  
        matrix = np.array([list(map(float, line.strip().split())) for line in file])  
    return matrix  
  
def pose_distance(matrix1, matrix2):  
    """ Calculate the Frobenius norm of the difference between two 4x4 pose matrices. """  
    difference = matrix1 - matrix2  
    distance = np.linalg.norm(difference)  
    return distance  
  
def compare_poses(folderA, folderB):  
    """ Compare a random pose from folder A to all poses in folder B and find the closest pose. """  
    # Get a random pose file from folder A  
    poses_folder_A = os.path.join(folderA, 'poses')  
    random_filename = random.choice(os.listdir(poses_folder_A))  
    pose_path_A = os.path.join(poses_folder_A, random_filename)  
    pose_matrix_A = read_pose_matrix(pose_path_A)  
  
    # Initialize variables to find the closest pose  
    min_distance = float('inf')  
    closest_pose = None  
  
    # Compare to all poses in folder B  
    poses_folder_B = os.path.join(folderB, 'poses')  
    for filename in os.listdir(poses_folder_B):  
        pose_path_B = os.path.join(poses_folder_B, filename)  
        pose_matrix_B = read_pose_matrix(pose_path_B)  
        distance = pose_distance(pose_matrix_A, pose_matrix_B)  
  
        if distance < min_distance:  
            min_distance = distance  
            closest_pose = filename  
  
    return random_filename, closest_pose, min_distance  
  
def display_images(folderA, folderB, filenameA, filenameB):  
    """ Display images side by side for comparison. """  
    img_path_A = os.path.join(folderA, 'rgb', filenameA.replace('.txt', '.png'))  
    img_path_B = os.path.join(folderB, 'rgb', filenameB.replace('.txt', '.png'))  
  
    img_A = mpimg.imread(img_path_A)  
    img_B = mpimg.imread(img_path_B)  
  
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  
    axs[0].imshow(img_A)  
    axs[0].set_title('Image from Folder A')  
    axs[0].axis('off')  
  
    axs[1].imshow(img_B)  
    axs[1].set_title('Closest Image from Folder B')  
    axs[1].axis('off')  
  
    plt.show()  
  
# Example usage  
folderA = '/home/elfar/datasets/dsanerf-objects-2'  
folderB = '/home/elfar/datasets/dsanerf-base-2'  
  
random_filename, closest_pose, distance = compare_poses(folderA, folderB)  
print(f"The closest pose in folder B is {closest_pose} with a distance of {distance}")  
  
# Display images  
display_images(folderA, folderB, random_filename, closest_pose)  
