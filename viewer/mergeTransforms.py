import json  
import os  
  
def load_json(file_path):  
    with open(file_path, 'r') as file:  
        data = json.load(file)  
    return data  
  
def merge_transforms(json_files, output_file):  
    merged_data = None  
      
    for json_file in json_files:  
        data = load_json(json_file)  
          
        if merged_data is None:  
            merged_data = data  
        else:  
            # Assuming all other data except 'frames' is the same and can be taken from the first file  
            merged_data['frames'].extend(data['frames'])  
      
    with open(output_file, 'w') as file:  
        json.dump(merged_data, file, indent=4)  
  
# List of transforms.json file paths to merge  
json_files = [  
    '/media/elfar/One Touch/l3_1/transforms.json',  
    '/media/elfar/One Touch/l3_2/transforms.json',  
    '/media/elfar/One Touch/l3_3/transforms.json',  
    '/media/elfar/One Touch/l3_4/transforms.json',  
    '/media/elfar/One Touch/l3_5/transforms.json',  
]  
  
# Output file path  
output_file = 'merged_transforms.json'  
  
# Perform merge  
merge_transforms(json_files, output_file)  
