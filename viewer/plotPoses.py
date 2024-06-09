import json
from sys import argv  
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
  
def load_transforms(file_path):  
    with open(file_path, 'r') as f:  
        data = json.load(f)  
    return data['frames']  
  
def extract_camera_positions_and_directions(frames):  
    positions = []  
    directions = []  
      
    for frame in frames:  
        transform = np.array(frame['transform_matrix'])  
        # Camera position is the translation part of the matrix  
        position = transform[:3, 3]  
        positions.append(position)  
          
        # Assuming the camera is looking along the negative z-axis of the camera's local coordinate system  
        # Camera direction can be found by transforming the negative z-axis by the rotation part of the matrix  
        rotation = transform[:3, :3]  
        direction = rotation @ np.array([0, 0, -1])  
        directions.append(direction)  
      
    return np.array(positions), np.array(directions)  
  
def plot_cameras(positions, directions):  
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')  
      
    ax.quiver(  
        positions[:, 0], positions[:, 1], positions[:, 2],  
        directions[:, 0], directions[:, 1], directions[:, 2],  
        length=0.1, normalize=True  
    )  
      
    ax.set_xlabel('X')  
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')  
    plt.title('Camera Positions and Viewing Directions')  
    plt.show()  
  
# Load transformations  
file_path = argv[1]  # Change to your actual file path  
frames = load_transforms(file_path)  
  
# Extract positions and directions  
positions, directions = extract_camera_positions_and_directions(frames)  
  
# Plot  
plot_cameras(positions, directions)  
