from annoy import AnnoyIndex  
import numpy as np  
import time  
  
def flatten_matrix(matrix):  
    """ Flatten the matrix to a vector. Exclude the last row as it is [0, 0, 0, 1] in homogeneous coordinates. """  
    return matrix[:3].flatten()  
  
def frobenius_norm(matrix1, matrix2):  
    """ Compute the Frobenius norm of the difference between two matrices. """  
    return np.linalg.norm(matrix1 - matrix2)  
  
# Create an Annoy index for vectors of length 12 (since we flatten the 4x3 part of the 4x4 matrix)  
t = AnnoyIndex(12, 'euclidean')  
  
# Assume matrices are stored in some list `matrices`  
matrices = [np.random.rand(4, 4) for _ in range(1000000)]  # Example matrices  
  
# Add items to the index  
for i, mat in enumerate(matrices):  
    v = flatten_matrix(mat)  
    t.add_item(i, v)  
  
# Build the index  
t.build(10)  # 10 trees  
  
# To find the nearest matrix to a query matrix  
query_matrix = np.random.rand(4, 4)  
query_vector = flatten_matrix(query_matrix)  
  
# Find the 100 approximate nearest neighbors  
candidate_ids = t.get_nns_by_vector(query_vector, 100)  
  
# Now refine the search by computing the exact Frobenius norm  
min_frob_norm = float('inf')  
closest_id = None  
  
start_refined_search_time = time.time()  
for idx in candidate_ids:  
    candidate_matrix = matrices[idx]  
    frob_norm = frobenius_norm(query_matrix, candidate_matrix)  
    if frob_norm < min_frob_norm:  
        min_frob_norm = frob_norm  
        closest_id = idx  
end_refined_search_time = time.time()  
  
refined_search_time = (end_refined_search_time - start_refined_search_time) * 1000  # Convert to milliseconds  
closest_matrix = matrices[closest_id]  
  
print("Time taken for refined search: {:.2f} ms".format(refined_search_time))  
print("Closest matrix ID with Frobenius norm:", closest_id)  
print("Closest matrix:\n", closest_matrix)  
