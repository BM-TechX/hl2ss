from annoy import AnnoyIndex  
import numpy as np  
import time  
  
def flatten_matrix(matrix):  
    """ Flatten the matrix to a vector. Exclude the last row as it is [0, 0, 0, 1] in homogeneous coordinates. """  
    return matrix[:3].flatten()  
  
# Create an Annoy index for vectors of length 12 (since we flatten the 4x3 part of the 4x4 matrix)  
t = AnnoyIndex(12, 'euclidean')  
  
# Assume matrices are stored in some list `matrices`  
matrices = [np.random.rand(4, 4) for _ in range(1000000)]  # Example matrices  
  
# Add items to the index  
for i, mat in enumerate(matrices):  
    v = flatten_matrix(mat)  
    t.add_item(i, v)  
  
# Build the index  
start_build_time = time.time()  
t.build(10)  # 10 trees  
end_build_time = time.time()  
  
build_time = (end_build_time - start_build_time) * 1000  # Convert to milliseconds  
print("Time taken to build index: {:.2f} ms".format(build_time))  
  
# To find the nearest matrix to a query matrix  
query_matrix = np.random.rand(4, 4)  
query_vector = flatten_matrix(query_matrix)  
  
start_search_time = time.time()  
nearest_id = t.get_nns_by_vector(query_vector, 1)[0]  
end_search_time = time.time()  
  
search_time = (end_search_time - start_search_time) * 1000  # Convert to milliseconds  
nearest_matrix = matrices[nearest_id]  
  
print("Time taken to find nearest matrix: {:.2f} ms".format(search_time))  
print("Nearest matrix ID:", nearest_id)  
print("Nearest matrix:\n", nearest_matrix)  
