'''
# Description
Randomly pick two points as the guess for the center of the graph. Then take the distance between 
each individual point and the guess location and assign them to the category with the lowest distance.
Take average of each group and move two guesses over to those locations. Eventually the guesses will
move to the centroid of each category.


# Category Assignment - points to cluster centroids
For x_i to x_m where m is the number of points/training-examples, lets say there are k centroids/categories.
then n_i to n_k are the centroid.
therefore,
c - list of centroid and there categories.
will be assgined ci = min_k|x_i - m_k|^2(means loop through k)
compute i to m




# Mean Computation - Move cluster centroids
for k = 1 to K:
    m_k = mean(all(ci = m_k)) (compute for each axis seperately)



#Optimisation

m_k = cluster centroid k
m_c_i = cluster centroid to which example x_i is assigned
c_i = index of cluster (according to m_k) to which x_i is currently assigned


J(c1, ..., c_m, m_c_i, ...., m_c_k) = 1/m * sum(x_i - m_c_i)^2


#Finding Global Minima

Initialize min_J() and m_K 

for i in range(k(any number)):
    Random initialize m_k_temp,
    Compute J_Temp()
    If J_Temp() < J():
        change initial values of m_k to m_k_temp

'''







import numpy as np
import matplotlib.pyplot as plt
from utils import *




def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """

    # Set K
    K = centroids.shape[0]
    L = X.shape[0]
    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)
    
    # Loop over each example in X
    for i in range(X.shape[0]):
        # Compute the squared distance between the i-th example and each centroid
        distances = np.sum((X[i] - centroids)**2, axis=1)
        
        # Find the index of the centroid with the smallest distance
        idx[i] = np.argmin(distances)
    
    return idx 
            
        
     ### END CODE HERE ###
    
    return idx


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    
    for i in range(K):
        total = np.zeros(X.shape[1])
        count = 0
        for j in range(m):
            if(idx[j] == i):
                total += X[j]
                count+=1
        total = total/count
        centroids[i] = total
    ### END CODE HERE ## 
    
    return centroids
def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

# Load an example dataset
X = np.array([[ 1.84207953,  4.6075716 ],
 [ 5.65858312 , 4.79996405],
 [ 6.35257892 , 3.2908545 ],
 [ 2.90401653 , 4.61220411],
 [ 3.23197916  ,4.93989405],
 [ 1.24792268 , 4.93267846],
 [ 1.97619886 , 4.43489674],
 [ 2.23454135  ,5.05547168],
 [ 2.02358978,  0.44771614],
 [ 3.62202931,  1.28643763],
 [ 2.42865879 , 0.86499285],
 [ 2.09517296  ,1.14010491],
 [ 5.29239452  ,0.36873298],
 [ 2.07291709  ,1.16763851],
 [ 0.94623208  ,0.24522253],
 [ 2.73911908  ,1.10072284],
 [ 6.00506534  ,2.72784171],
 [ 6.05696411  ,2.94970433],
 [ 6.77012767  ,3.21411422],
 [ 5.64034678  ,2.69385282],
 [ 5.63325403  ,2.99002339],
 [ 6.17443157 , 3.29026488],
 [ 7.24694794,  2.96877424]])

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])

# Number of iterations
max_iters = 10

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
