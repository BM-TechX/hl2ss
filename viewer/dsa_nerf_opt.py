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
folders = ['rgb', 'poses', 'calibration']  
for folder in folders:  
    os.makedirs(folder, exist_ok=True)  
  
# Helper functions  
def calculate_motion_blur_score(image):  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)  
    return laplacian.var() / 1000.0  
  
def save_frame_data(data, timestamp):  
    if data.payload.image is not None:  
        image_filename = f"./rgb/frame_{timestamp}.png"  
        cv2.imwrite(image_filename, data.payload.image)  
    pose_filename = f"./poses/frame_{timestamp}.txt"  
    with open(pose_filename, 'w') as pose_file:  
        for row in data.pose.T:  
            pose_file.write(" ".join([str(value) for value in row]) + "\n")  
    return image_filename, pose_filename  
  
def frame_processing_thread(data, frame_count):  
    timestamp = time.time()  # Using time since epoch as unique identifier  
    blur = calculate_motion_blur_score(data.payload.image)  
    if blur >= 0.10:  
        image_filename, pose_filename = save_frame_data(data, timestamp)  
        return {  
            "timestamp": timestamp,  
            "blur": blur,  
            "image_filename": image_filename,  
            "pose_filename": pose_filename  
        }  
    return None  
  
# Start video stream  
client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height,  
                         framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)  
client.open()  
  
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
        frame_count += 1  
        t = threading.Thread(target=frame_processing_thread, args=(data, frame_count))  
        t.start()  
        process_threads.append(t)  
  
# Cleanup  
for t in process_threads:  
    t.join()  
    if t.result:  
        results.append(t.result)  
  
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
    "frames": results
}
  
#nerfstudio_json["frames"] = frames_data  
with open('dsanerf/transforms.json', 'w') as json_file:  
    json.dump(nerfstudio_json, json_file, indent=4)
    client.close()  
  
client.close()  
listener.join()  
hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)  
