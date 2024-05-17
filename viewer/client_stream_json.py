# ------------------------------------------------------------------------------
# This script receives video from the HoloLens front RGB camera and plays it.
# The camera supports various resolutions and framerates. See
# https://github.com/jdibenes/hl2ss/blob/main/etc/pv_configurations.txt
# for a list of supported formats. The default configuration is 1080p 30 FPS.
# The stream supports three operating modes: 0) video, 1) video + camera pose,
# 2) query calibration (single transfer).
# Press esc to stop.
# ------------------------------------------------------------------------------

from pynput import keyboard

import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import json
import numpy as np

# Settings --------------------------------------------------------------------

# HoloLens address
host = "10.9.50.22"

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Enable Mixed Reality Capture (Holograms)
enable_mrc = False

# Camera parameters
width = 1920
height = 1080
framerate = 15

# Framerate denominator (must be > 0)
# Effective FPS is framerate / divisor
divisor = 1

# Video encoding profile
profile = hl2ss.VideoProfile.H265_MAIN

# Decoded format
# Options include:
# 'bgr24'
# 'rgb24'
# 'bgra'
# 'rgba'
# 'gray8'
decoded_format = 'bgr24'

# ------------------------------------------------------------------------------

def calculate_motion_blur_score(image):  
    # Convert to grayscale  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
      
    # Apply edge detection (e.g., using the Sobel operator)  
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  
      
    # Calculate the magnitude of the gradients  
    magnitude = np.sqrt(sobelx**2 + sobely**2)  
      
    # Threshold the magnitude image to create a binary edge image  
    _, edge_image = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)  
      
    # Analyze the edges to determine the blur extent (simple example using edge width)  
    # More sophisticated methods like FFT analysis can also be used here  
    blur_extent = np.mean(edge_image)  
      
    # Normalize the score based on your application's requirements  
    # Lower scores might indicate more blur  
    motion_blur_score = 1 - blur_extent / 255  
      
    return motion_blur_score  

def get_velocity(T1, T2, dt):  
    """  
    Calculates the angular and linear velocity from two transformation matrices and time difference.  
    Args:  
        T1: A 4x4 numpy array representing the first transformation matrix.  
        T2: A 4x4 numpy array representing the second transformation matrix.  
        dt: The time difference between the frames captured by T1 and T2 (in seconds).  
    Returns:  
        A tuple containing two numpy arrays:  
            - angular_velocity: The angular velocity vector (rad/s) in the world frame.  
            - linear_velocity: The linear velocity vector (m/s) in the world frame.  
    """  
  
    # Extract rotation matrices  
    R1 = T1[:3, :3]  
    R2 = T2[:3, :3]  
  
    # Calculate relative rotation matrix  
    relative_rotation_matrix = np.dot(R2, R1.T)  
  
    # Convert rotation matrix to axis-angle representation  
    # Rodrigues' rotation formula  
    r, _ = cv2.Rodrigues(relative_rotation_matrix)  
  
    # Angular velocity (axis-angle scaled by 1/dt)  
    angular_velocity = r.flatten() / dt  
  
    # Extract translation vectors - ther first 3 numbers of the last row
    translation1 = T1[:3, 3]
    translation2 = T2[:3, 3]
        
    # Calculate linear velocity  
    linear_velocity = (translation2 - translation1) / dt 
  
    #print("Translation1:", translation1)  
    #print("Translation2:", translation2)  
    #print("Linear Velocity Pre-Return:", linear_velocity)  
      
    return angular_velocity, linear_velocity  


hl2ss_lnm.start_subsystem_pv(
    host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc)

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_pv(
        host, hl2ss.StreamPort.PERSONAL_VIDEO, width, height, framerate)
    print('Calibration')
    print(f'Focal length: {data.focal_length}')
    print(f'Principal point: {data.principal_point}')
    print(f'Radial distortion: {data.radial_distortion}')
    print(f'Tangential distortion: {data.tangential_distortion}')
    print('Projection')
    print(data.projection)
    print('Intrinsics')
    print(data.intrinsics)
    print('RigNode Extrinsics')
    print(data.extrinsics)
else:
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height,
                             framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)
    client.open()

    prev_pose = None
    cx = 0.0
    cy = 0.0
    fl_x = 0.0
    fl_y = 0.0

    # Define a list to store frame data
    frames_data = []

    while (enable):

        data = client.get_next_packet()

        if data is None:
            continue

        if prev_pose is None:
            prev_pose = data.pose.T
            continue
        else:
            print(f'Pose at time {data.timestamp}')
            print(data.pose.T)
            print(f'Focal length: {data.payload.focal_length}')
            print(f'Principal point: {data.payload.principal_point}')
            
            cx = data.payload.principal_point[0]
            cy = data.payload.principal_point[1]
            fl_x = data.payload.focal_length[0]
            fl_y = data.payload.focal_length[1]

            angular_velocity, linear_velocity = get_velocity(
                data.pose.T, prev_pose, 1/framerate)

            prev_pose = data.pose.T
            print("Angular velocity:", angular_velocity)
            print("Linear velocity:", linear_velocity)
            
            blur = calculate_motion_blur_score(data.payload.image)

            # Collect the pose information and other relevant data
            frame_data = {
                # Placeholder, replace with actual data if available
                "camera_angular_velocity": angular_velocity.tolist(),
                # Placeholder, replace with actual data if available
                "camera_linear_velocity": linear_velocity.tolist(),
                # Assuming timestamp can be used as a unique identifier
                "file_path": f"./images/frame_{data.timestamp}.jpg",
                "motion_blur_score": float(blur),  # Placeholder, replace with actual data if available
                "transform_matrix": data.pose.T.tolist()  # Assuming this is the correct format
            }

            # Save the image frame to a file
            image_filename = f"./test/images/frame_{data.timestamp}.jpg"
            cv2.imwrite(image_filename, data.payload.image)

            # Add the frame data to the list
            frames_data.append(frame_data)

            cv2.imshow('Video', data.payload.image)
            if cv2.waitKey(1) == 27:  # Check for ESC key
                break

        cv2.imshow('Video', data.payload.image)
        cv2.waitKey(1)

    client.close()
    listener.join()

# Define the rest of the JSON structure
nerfstudio_json = {
    "h": 1080,
    "k1": 0,
    "k2": 0,
    "orientation_override": "none",
    "p1": 0,
    "p2": 0,
    #"ply_file_path": "./sparse_pc.ply",
    "w": 1920,
    "aabb_scale": 16,
    "auto_scale_poses_override": False,
    # Assuming this is the correct value
    "cx": float(cx),
    # Assuming this is the correct value
    "cy": float(cy),
    "fl_x": float(fl_x),  # Assuming this is the correct value
    "fl_y": float(fl_y),  # Assuming this is the correct value
    "frames": frames_data
}

# Save the collected data to a JSON file
with open('test/transforms.json', 'w') as json_file:
    json.dump(nerfstudio_json, json_file, indent=4)

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
