import json  
import cv2  
import numpy as np  
import threading  
from pynput import keyboard  
import hl2ss_imshow  
import hl2ss  
import hl2ss_lnm  
import time  
import os  
  
# Settings  
host = "10.9.50.22"  
mode = hl2ss.StreamMode.MODE_1  
enable_mrc = False  
width = 1920  
height = 1080  
framerate = 30  
divisor = 1  
profile = hl2ss.VideoProfile.H265_MAIN  
decoded_format = 'bgr24'  
  
# Folder setup  
folders = ['dsanerf/rgb', 'dsanerf/poses', 'dsanerf/calibration']  
for folder in folders:  
    os.makedirs(folder, exist_ok=True)  
  
# Helper functions  
def calculate_motion_blur_score(image):  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)  
    return laplacian.var() / 1000.0

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
 
  
def save_frame_data(data, timestamp):  
    if data.payload.image is not None:  
        image_filename = f"./dsanerf/rgb/frame_{timestamp}.png"  
        cv2.imwrite(image_filename, data.payload.image)  
    pose_filename = f"./dsanerf/poses/frame_{timestamp}.txt"  
    with open(pose_filename, 'w') as pose_file:  
        for row in data.pose.T:  
            pose_file.write(" ".join([str(value) for value in row]) + "\n")
        
    f_p = data.payload.focal_length
    p_p = data.payload.principal_point
    
    #Euclidian distance from focal point to principal point
    focal_length = np.linalg.norm(f_p - p_p)

    image_filename = f"./dsanerf/calibration/frame_{timestamp}.txt"
    with open(image_filename, 'w') as calibration_file:
        calibration_file.write(str(focal_length))
  
    return image_filename, pose_filename  
  
def frame_processing_thread(data, frame_count):  
    timestamp = time.time()  # Using time since epoch as unique identifier  
    blur = calculate_motion_blur_score(data.payload.image)  
    if blur >= 0.10:  
        image_filename, pose_filename = save_frame_data(data, timestamp)  
        transform_matrix = data.pose.T.tolist()  
        return {  
            "camera_angular_velocity": get_velocity(data.pose.T, prev_pose, 1/framerate)[0].tolist(),  
            "camera_linear_velocity": get_velocity(data.pose.T, prev_pose, 1/framerate)[1].tolist(),  
            "file_path": image_filename,  
            "motion_blur_score": blur,  
            "transform_matrix": transform_matrix  
        }  
    return None  
  
# Start video stream  
client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height,  
                         framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)  
client.open()  
prev_pose = None  
  
# Main loop  
enable = True  
frame_count = 0  
process_threads = []  
results = []  
  
def on_press(key):  
    global enable  
    if key == keyboard.Key.esc:  
        enable = False  
  
listener = keyboard.Listener(on_press=on_press)  
listener.start()  
  
while enable:  
    data = client.get_next_packet()  
    if data and data.payload and data.payload.image.size != 0:  
        if prev_pose is None:  
            prev_pose = data.pose.T  
        frame_count += 1  
        t = threading.Thread(target=frame_processing_thread, args=(data, frame_count))  
        t.start()  
        process_threads.append(t)  
  
# Cleanup  
for t in process_threads:  
    t.join()  
    if t.result:  
        results.append(t.result)  
  
# Save results to JSON  
nerfstudio_json = {  
    "h": height,  
    "k1": 0,  # Assuming no radial distortion coefficient k1  
    "k2": 0,  # Assuming no radial distortion coefficient k2  
    "orientation_override": "none",  
    "p1": 0,  # Assuming no tangential distortion coefficient p1  
    "p2": 0,  # Assuming no tangential distortion coefficient p2  
    "w": width,  
    "aabb_scale": 16,  
    "auto_scale_poses_override": False,  
    "cx": data.payload.principal_point[0],  # Assuming last packet's cx is representative  
    "cy": data.payload.principal_point[1],  # Assuming last packet's cy is representative  
    "fl_x": data.payload.focal_length[0],  # Assuming last packet's fl_x is representative  
    "fl_y": data.payload.focal_length[1],  # Assuming last packet's fl_y is representative  
    "frames": results  
}  
  
# Write the Nerfstudio JSON structure to a file  
with open('dsanerf/transforms.json', 'w') as json_file:  
    json.dump(nerfstudio_json, json_file, indent=4)  
  
client.close()  
listener.join()  
hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)  
  
print("Data capture and saving complete. JSON structure saved to 'dsanerf/transforms.json'.")  
