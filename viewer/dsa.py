import cv2  
import numpy as np  
import os  
from datetime import datetime  
import multiprocessing as mp  
from pynput import keyboard  
import open3d as o3d  
  
import hl2ss  
import hl2ss_lnm  
import hl2ss_mp 
import hl2ss_utilities 
  
# Settings  
host = '10.9.50.22'  # HoloLens address  
calibration_path = '../calibration'  # Calibration path  
pv_width, pv_height, pv_fps = 640, 360, 30  # Front RGB camera parameters  
buffer_size = 10  # Buffer length in seconds  
max_depth = 3.0  # Maximum depth in meters  
  
def main():  
    dataset_dir = 'dsa'  
    setup_directories(dataset_dir)  
    save_intrinsic_calibration(dataset_dir)  
    setup_and_stream(host, dataset_dir, pv_width, pv_height, pv_fps, buffer_size)  
  
def setup_directories(dataset_dir):  
    os.makedirs(os.path.join(dataset_dir, 'rgb'), exist_ok=True)  
    os.makedirs(os.path.join(dataset_dir, 'depth'), exist_ok=True)  
    os.makedirs(os.path.join(dataset_dir, 'poses'), exist_ok=True)  
    os.makedirs(os.path.join(dataset_dir, 'calibration'), exist_ok=True)  
  
def save_intrinsic_calibration(dataset_dir):  
    np.savetxt(os.path.join(dataset_dir, 'calibration', 'intrinsic.txt'), np.eye(3))  
  
def setup_and_stream(host, dataset_dir, width, height, fps, buffer_size):  
    listener = keyboard.Listener(on_press=on_key_press)  
    listener.start()  
  
    producer, consumer, sink_pv, sink_depth = initialize_streaming(host, width, height, fps, buffer_size)  
  
    try:  
        process_frames(consumer, dataset_dir, width, height, listener, sink_pv, sink_depth)  
    finally:  
        cleanup(listener, producer, consumer, sink_pv, sink_depth)  
  
def on_key_press(key):  
    if key == keyboard.Key.space:  
        return False  # Returning False stops the listener  
  
def initialize_streaming(host, width, height, fps, buffer_size):  
    producer = hl2ss_mp.producer()  
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=width, height=height, framerate=fps))  
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))  
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, fps * buffer_size)  
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_size)  
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)  
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)  
  
    consumer = hl2ss_mp.consumer()  
    manager = mp.Manager()  
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)
    
    sink_pv.get_attach_response()
    sink_depth.get_attach_response()

    # Initialize PV intrinsics and extrinsics ---------------------------------
    #pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    #pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    VI = hl2ss_utilities.framerate_counter()
    VI.reset()

  
    return producer, consumer, sink_pv, sink_depth  # Make sure to return the correct number of items  
  
def process_frames(consumer, dataset_dir, width, height, listener, sink_pv, sink_depth):  
    while listener.running:  
        #frame_pv = sink_pv.wait_for_frame()  
        #frame_depth = sink_depth.wait_for_frame() 
        sink_depth.acquire()
                # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
        _, frame_depth = sink_depth.get_most_recent_frame()
        if ((frame_depth is None) or (not hl2ss.is_valid_pose(frame_depth.pose))):
            continue

        _, frame_pv = sink_pv.get_nearest(frame_depth.timestamp)
        if ((frame_pv is None) or (not hl2ss.is_valid_pose(frame_pv.pose))):
            continue
 
  
        if frame_pv is None or frame_depth is None:  
            continue  # Skip if any frame is missing  
  
        rgb_image = frame_pv.payload.image  # Access the RGB image  
        depth_image = frame_depth.payload.depth  # Access the depth data  
        pose = frame_depth.payload.pose  # Access the pose data  
  
        # Process and save data  
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')  
        save_data(rgb_image, depth_image, pose, dataset_dir, timestamp)  
  
def save_data(rgb_image, depth_image, pose, dataset_dir, timestamp):  
    cv2.imwrite(os.path.join(dataset_dir, 'rgb', f'{timestamp}.png'), rgb_image)  
    np.save(os.path.join(dataset_dir, 'depth', f'{timestamp}.npy'), depth_image)  # Save depth as numpy array  
    np.savetxt(os.path.join(dataset_dir, 'poses', f'{timestamp}.txt'), pose)  
  
def cleanup(listener, producer, consumer, sink_pv, sink_depth): 
    #sink_pv.detach()
    #sink_depth.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
     
    #listener.stop()  
    #producer.stop_all()  # Stop all streams  
    #consumer.close_all_sinks()  # Close all sinks  
  
if __name__ == '__main__':  
    main()  
