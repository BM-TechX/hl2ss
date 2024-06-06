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
import argparse  
import shutil   
  
# Parse command-line arguments  
parser = argparse.ArgumentParser(description='Process frames and save data.')  
parser.add_argument('folder', type=str, help='Directory to store the outputs')  
args = parser.parse_args()  

# Global counter for frames that passed the blur filter  
frames_passed_blur_filter = 0  
lock = threading.Lock()  # Lock for thread-safe increments to the counter  
  
# Settings  
host = "192.168.0.168"
mode = hl2ss.StreamMode.MODE_1  
enable_mrc = False  
width = 1920  
height = 1080  
framerate = 15  
divisor = 1  
profile = hl2ss.VideoProfile.H265_MAIN  
decoded_format = 'bgr24'
  
# Folder setup  
base_folder = args.folder  
folders = [f'{base_folder}/rgb', f'{base_folder}/poses', f'{base_folder}/calibration']  
transforms_file = f"{base_folder}/transforms.json"  
  
# Remove existing files and folders if they exist  
if os.path.exists(transforms_file):  
    os.remove(transforms_file)  
  
for folder in folders:  
    if os.path.exists(folder):  
        shutil.rmtree(folder)  
    os.makedirs(folder, exist_ok=True)  
        
hl2ss_lnm.start_subsystem_pv(
    host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc)
  
# Helper functions
def print_frame_count_every_3_seconds():  
    global frames_passed_blur_filter  
    last_count = 0  
  
    while enable:  # Assuming 'enable' is your main loop control variable  
        with lock:  
            new_frames = frames_passed_blur_filter - last_count  
            print(f"Frames passed blur filter in the last 3 seconds: {new_frames}")  
            last_count = frames_passed_blur_filter  
        time.sleep(3)  
  
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
    no_dot_timestamp = formatted_timestamp.replace(".", "-")
  
    if data.payload.image is not None:  
        image_filename = f"./{base_folder}/rgb/frame_{no_dot_timestamp}.png"  
        cv2.imwrite(image_filename, data.payload.image)  
    pose_filename = f"./{base_folder}/poses/frame_{no_dot_timestamp}.txt"  
    with open(pose_filename, 'w') as pose_file:  
        for row in data.pose.T:  
            pose_file.write(" ".join([str(value) for value in row]) + "\n")
        
    f_p = data.payload.focal_length
    p_p = data.payload.principal_point
    
    #Euclidian distance from focal point to principal point
    focal_length = np.linalg.norm(f_p - p_p)

    cal_filename = f"./{base_folder}/calibration/frame_{no_dot_timestamp}.txt"
    with open(cal_filename, 'w') as calibration_file:
        calibration_file.write(str(focal_length))
        
    image_filename = f"./rgb/frame_{no_dot_timestamp}.png"
  
    return image_filename  
  

# Helper function to handle thread results  
def handle_thread_result(result, results_list):  
    if result is not None:  
        results_list.append(result)  
  
def frame_processing_thread(data, results_list, prev_pose):  
    global frames_passed_blur_filter  
  
    timestamp = time.time()  
    blur = calculate_motion_blur_score(data.payload.image)  
    if blur >= 0.10:  
        with lock:  
            frames_passed_blur_filter += 1  
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
                
# Start video stream  
client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)  
client.open()  
prev_pose = None  
  
# Main loop  
enable = True  
threads = []  
results = []  
  
def on_press(key):  
    global enable  
    enable = key != keyboard.Key.space  
    return enable  
  
listener = keyboard.Listener(on_press=on_press)  
listener.start()  

# Start the frame count printer thread  
frame_count_thread = threading.Thread(target=print_frame_count_every_3_seconds)  
frame_count_thread.start()  
  
while enable:  
    data = client.get_next_packet()  
    if data and data.payload and getattr(data.payload, 'image', None) is not None and data.payload.image.size != 0:  
        current_pose = data.pose.T  
        if prev_pose is not None:  
            thread = threading.Thread(target=frame_processing_thread, args=(data, results, prev_pose))  
            thread.start()  
            threads.append(thread)  
        prev_pose = current_pose  # Update prev_pose after confirming the thread is created  
  
# Wait for all threads to complete  
for thread in threads:  
    thread.join()  
  
client.close()  
listener.join()
frame_count_thread.join()  
  
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
with open(f"{base_folder}/transforms.json", 'w') as json_file:  
    json.dump(nerfstudio_json, json_file, indent=4)  
  
hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)  
print("Finished processing frames.")  
   