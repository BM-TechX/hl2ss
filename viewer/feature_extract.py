import time
import numpy as np  
import tensorflow as tf  
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  
from tensorflow.keras.preprocessing import image  
from tensorflow.keras.models import Model  
from annoy import AnnoyIndex  
import glob  
import os  
  
# Load a pre-trained VGG16 model without the top classification layer  
base_model = VGG16(weights='imagenet', include_top=False)  
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)  
  
# Function to preprocess and extract features from an image  
def extract_features(img_path, model):  
    img = image.load_img(img_path, target_size=(224, 224))  
    x = image.img_to_array(img)  
    x = np.expand_dims(x, axis=0)  
    x = preprocess_input(x)  
    features = model.predict(x)  
    return features.flatten()  
  
# Precompute features for all images and save them  
def precompute_features(image_paths, model, feature_store_path):  
    features_dict = {}  
    for img_path in image_paths:  
        features = extract_features(img_path, model)  
        features_dict[img_path] = features  
    np.save(feature_store_path, features_dict)  
  
# Load precomputed features  
def load_features(feature_store_path):  
    return np.load(feature_store_path, allow_pickle=True).item()  
  
def create_ann_index(features_dict, vector_length=25088, n_trees=10):  
    index = AnnoyIndex(vector_length, 'angular')  
    for i, features in enumerate(features_dict.values()):  
        index.add_item(i, features)  
    index.build(n_trees)  
    index.save('features.ann')  
    return index  
  
def load_ann_index(path, vector_length=25088):  
    if os.path.exists(path):  
        index = AnnoyIndex(vector_length, 'angular')  
        index.load(path)  
        return index  
    else:  
        print(f"Failed to load Annoy index from {path}")  
        return None  
      
# Find the most similar image using Annoy  
def find_similar_images_ann(query_features, index, features_dict, n_neighbors=1):  
    nearest_ids = index.get_nns_by_vector(query_features, n_neighbors, include_distances=True)  
    return [(list(features_dict.keys())[i], dist) for i, dist in zip(*nearest_ids)]  
  
if __name__ == "__main__":  
    nerf_model_images_dir = '/home/elfar/software/nerfstudio/data/nerfstudio/test/images/'  
    feature_store_path = 'precomputed_features.npy' 
    features_dict = {}
    index = None
     
    nerf_model_images = glob.glob(os.path.join(nerf_model_images_dir, '*.jpg'))  
  
    if not os.path.exists(feature_store_path):  
        precompute_features(nerf_model_images, model, feature_store_path)
        #features_dict = load_features(feature_store_path)  
        #create_ann_index(features_dict, 25088)  # Adjust vector_length based on your model output  
    #else:  
    
    features_dict = load_features(feature_store_path)  
    query_img_path = '/home/elfar/software/nerfstudio/data/nerfstudio/test/images/frame_112398607108.jpg'  
    #query_features = extract_features(query_img_path, model)  
    #best_matching_image, similarity_score = find_similar_images_ann(query_features, index, features_dict, n_neighbors=1)[0]  
    
    if not os.path.exists('features.ann'):  
        print("Index file not found, creating index.")  
        index = create_ann_index(features_dict, 25088)  
    else:  
        index = load_ann_index('features.ann', 25088)  

    if index is None:  
        print("Error: Annoy index could not be created or loaded.")  
    else:  
        #Calculate how long it takes to find the most similar image. In milliseconds
        start_time = time.perf_counter()  
        query_features = extract_features(query_img_path, model)  
        results = find_similar_images_ann(query_features, index, features_dict, n_neighbors=1)
        # End timer  
        end_time = time.perf_counter()  
        
        # Time elapsed in milliseconds  
        elapsed_time_ms = (end_time - start_time) * 1000  
        
        print(f"The code took {elapsed_time_ms:.2f} milliseconds to execute.") 
         
        if results:  
            print(results)
            best_matching_image, similarity_score = results[0]  
            print(f"Best matching image: {best_matching_image}, Similarity score: {similarity_score}")  
        else:  
            print("No similar images found.")  
    #index = load_ann_index('features.ann', 25088)  # Correct vector length used  
  