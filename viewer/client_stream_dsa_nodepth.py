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
            
            focal_length = data.payload.focal_length[0]
            #Write the focal length to calibration/frameX.txt file
            calibration_filename = f"./dsa/calibration/frame_{data.timestamp}.txt"
            with open(calibration_filename, 'w') as calibration_file:
                calibration_file.write(str(focal_length))

            # Save the image frame to a file
            #image_filename = f"./dsa/rgb/frame_{data.timestamp}.jpg"
            #cv2.imwrite(image_filename, data.payload.image)
            
            image_filename = f"./dsa/rgb/frame_{data.timestamp}.png"
            cv2.imwrite(image_filename, data.payload.image)  
            
            transform_matrix = data.pose.T.tolist()
            pose_filename = f"./dsa/poses/frame_{data.timestamp}.txt"
            # Save the pose data to a file - in 4 lines like this
            #0.4895218312740326 0.06232452392578125 0.8697609305381775 -39.36280822753906
            #-0.8719829320907593 0.030700458213686943 0.48857253789901733 -17.508554458618164
            #0.00374798895791173 -0.9975836277008057 0.0693744644522667 1.766454815864563
            #0.0 0.0 0.0 1.0
            with open(pose_filename, 'w') as pose_file:
                for row in transform_matrix:
                    pose_file.write(" ".join([str(value) for value in row]) + "\n")

            cv2.imshow('Video', data.payload.image)
            if cv2.waitKey(1) == 27:  # Check for ESC key
                break

        cv2.imshow('Video', data.payload.image)
        cv2.waitKey(1)

    client.close()
    listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
