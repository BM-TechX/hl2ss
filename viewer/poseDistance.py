import numpy as np  
import os  
  
def read_pose_matrix(filepath):  
    """  
    Read a pose matrix from a text file.  
      
    Args:  
    filepath (str): Path to the text file containing the pose matrix.  
      
    Returns:  
    np.array: A 4x4 pose matrix.  
    """  
    with open(filepath, 'r') as file:  
        matrix = np.array([list(map(float, line.strip().split())) for line in file])  
    return matrix  
  
def pose_distance(matrix1, matrix2):  
    """  
    Calculate the Frobenius norm of the difference between two 4x4 pose matrices.  
    """  
    difference = matrix1 - matrix2  
    distance = np.linalg.norm(difference)  
    return distance  
  
def compare_poses(folderA, folderB, frameX):  
    """  
    Compare a specific pose from folder A to all poses in folder B and find the closest pose.  
      
    Args:  
    folderA (str): Directory containing poses for folder A.  
    folderB (str): Directory containing poses for folder B.  
    frameX (str): The frame identifier for the specific pose in folder A.  
      
    Returns:  
    str: Filename of the pose in folder B that is closest to the specified pose in folder A.  
    float: The distance to the closest pose.  
    """  
    # Path to the pose file in folder A  
    pose_path_A = os.path.join(folderA, 'poses', f'frame_{frameX}.txt')  
    pose_matrix_A = read_pose_matrix(pose_path_A)  
      
    # Initialize variables to find the closest pose  
    min_distance = float('inf')  
    closest_pose = None  
      
    # Compare to all poses in folder B  
    for filename in os.listdir(os.path.join(folderB, 'poses')):  
        pose_path_B = os.path.join(folderB, 'poses', filename)  
        pose_matrix_B = read_pose_matrix(pose_path_B)  
        distance = pose_distance(pose_matrix_A, pose_matrix_B)  
          
        if distance < min_distance:  
            min_distance = distance  
            closest_pose = filename  
      
    return closest_pose, min_distance  
  
# Example usage  
folderA = 'dsanerf-objects'  
folderB = 'dsanerf-base'  
frameX = '38021623153'  
frameX = '37752400506'
frameX='38198883505'
closest_pose, distance = compare_poses(folderA, folderB, frameX)  
print(f"The closest pose in folder B is {closest_pose} with a distance of {distance}")  
