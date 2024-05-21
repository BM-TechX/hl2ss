import os  
import shutil  
from random import shuffle  
  
def create_dir_if_not_exists(directory):  
    if not os.path.exists(directory):  
        os.makedirs(directory)  
  
def split_data(source_dir, train_dir, test_dir, split_ratio):  
    files = os.listdir(source_dir)  
    shuffle(files)  
      
    split_index = int(len(files) * split_ratio)  
    train_files = files[:split_index]  
    test_files = files[split_index:]  
      
    for f in train_files:  
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))  
      
    for f in test_files:  
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))  
  
# Define source directories  
base_dir = './dsanerf'  
calibration_dir = os.path.join(base_dir, 'calibration')  
poses_dir = os.path.join(base_dir, 'poses')  
rgb_dir = os.path.join(base_dir, 'rgb')  
  
# Define destination directories  
destination_dir = 'cantine'  
  
# Create train and test directories for each folder  
subfolders = ['calibration', 'poses', 'rgb']  
for subfolder in subfolders:  
    train_subfolder = os.path.join(destination_dir, 'train', subfolder)  
    test_subfolder = os.path.join(destination_dir, 'test', subfolder)  
    create_dir_if_not_exists(train_subfolder)  
    create_dir_if_not_exists(test_subfolder)  
  
# Split data  
split_ratio = 0.8  # 80% for training, 20% for testing  
  
split_data(calibration_dir, os.path.join(destination_dir, 'train', 'calibration'),   
           os.path.join(destination_dir, 'test', 'calibration'), split_ratio)  
split_data(poses_dir, os.path.join(destination_dir, 'train', 'poses'),   
           os.path.join(destination_dir, 'test', 'poses'), split_ratio)  
split_data(rgb_dir, os.path.join(destination_dir, 'train', 'rgb'),   
           os.path.join(destination_dir, 'test', 'rgb'), split_ratio)  
  
print("Data has been split into train and test sets.")  
