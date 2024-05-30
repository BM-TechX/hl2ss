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
from functools import partial 
  
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
    formatted_timestamp = f"{timestamp:.6f}"  
    if data.payload.image is not None:  
        image_filename = f"./dsanerf/rgb/frame_{formatted_timestamp}.png"  
        cv2.imwrite(image_filename, data.payload.image)  
    pose_filename = f"./dsanerf/poses/frame_{formatted_timestamp}.txt"  
    with open(pose_filename, 'w') as pose_file:  
        for row in data.pose.T:  
            pose_file.write(" ".join([str(value) for value in row]) + "\n")
        
    f_p = data.payload.focal_length
    p_p = data.payload.principal_point
    
    #Euclidian distance from focal point to principal point
    focal_length = np.linalg.norm(f_p - p_p)

    image_filename = f"./dsanerf/calibration/frame_{formatted_timestamp}.txt"
    with open(image_filename, 'w') as calibration_file:
        calibration_file.write(str(focal_length))
  
    return image_filename  
  

# Helper function to handle thread results  
def handle_thread_result(result, results_list):  
    if result is not None:  
        results_list.append(result)  
  
# Frame processing with threading  
def frame_processing_thread(data, results_list):  
    timestamp = time.time()  
    blur = calculate_motion_blur_score(data.payload.image)  
    if blur >= 0.10:  
        image_filename = save_frame_data(data, timestamp)  
        transform_matrix = data.pose.T.tolist()  
        angular_velocity, linear_velocity = get_velocity(data.pose.T, prev_pose, 1/framerate)  
        result = {  
            "camera_angular_velocity": angular_velocity.tolist(),  
            "camera_linear_velocity": linear_velocity.tolist(),  
            "file_path": image_filename,  
            "motion_blur_score": blur,  
            "transform_matrix": transform_matrix  
        }  
        handle_thread_result(result, results_list)  
    else:  
        handle_thread_result(None, results_list)  
        
'''
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
'''

# Start video stream  
client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height,  
                         framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)  
client.open()  
prev_pose = None  
  
# Main loop  
enable = True  
process_threads = []  
results = [] 
threads = []  

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable
    
listener = keyboard.Listener(on_press=on_press)  
listener.start()  
  
while enable:  
    data = client.get_next_packet()  
    if data and data.payload and data.payload.image.size != 0:  
        if prev_pose is None:  
            prev_pose = data.pose.T  
        thread = threading.Thread(target=frame_processing_thread, args=(data, results))  
        thread.start()  
        threads.append(thread)  
  
# Wait for all threads to complete  
for thread in threads:  
    thread.join()  
  
client.close()  
listener.join()  

# Now results should contain all the data from the threads  
nerfstudio_json = {  
    "h": height,  
    "k1": 0,  
    "k2": 0,  
    "orientation_override": "none",  
    "p1": 0,  
    "p2": 0,  
    "w": width,  
    "aabb_scale": 16,  
    "auto_scale_poses_override": False,  
    "cx": float(data.payload.principal_point[0]),  
    "cy": float(data.payload.principal_point[1]),  
    "fl_x": float(data.payload.focal_length[0]),  
    "fl_y": float(data.payload.focal_length[1]),  
    "frames": results  
}  
  
# Write the Nerfstudio JSON structure to a file  
with open('dsanerf/transforms.json', 'w') as json_file:  
    json.dump(nerfstudio_json, json_file, indent=4)  
  
hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)  
print("Finished processing frames.")