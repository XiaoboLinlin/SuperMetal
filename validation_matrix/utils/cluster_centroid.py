import numpy as np

def find_centroid(coordinates, clusters):
    # Find the unique cluster labels
    unique_clusters = np.unique(clusters)
    # Initialize a dictionary to store centroids
    centroids = []

    # Calculate the centroid for each cluster
    for cluster in unique_clusters:
        if cluster != -1:  # Exclude noise points
            points_in_cluster = coordinates[clusters == cluster]
            centroid = points_in_cluster.mean(axis=0)
            centroids.append(centroid)
    return np.array(centroids)


# import torch

# def find_centroid(coordinates, clusters):
#     """
#     Calculate centroids for each cluster using PyTorch tensors.
    
#     :param coordinates: PyTorch tensor of coordinates.
#     :param clusters: PyTorch tensor of cluster labels.
#     :return: PyTorch tensor of centroids.
#     """
#     # Find the unique cluster labels
#     unique_clusters = torch.unique(clusters)
#     centroids = []

#     # Calculate the centroid for each cluster
#     for cluster in unique_clusters:
#         if cluster.item() != -1:  # Exclude noise points
#             mask = clusters == cluster
#             points_in_cluster = coordinates[mask]
#             centroid = points_in_cluster.mean(dim=0)
#             centroids.append(centroid)

#     return torch.stack(centroids)

# # Example usage
# # Assuming coordinates and clusters are PyTorch tensors
# coordinates = torch.tensor([
#     [1.0, 2.0, 1.0], [1.0, 2.0, 2.0], [2.0, 2.0, 2.0],
#     [8.0, 8.0, 8.0], [8.0, 9.0, 8.0], [9.0, 8.0, 9.0],
#     [100.0, 100.0, 101.0], [100.0, 101.0, 100.0], [101.0, 100.0, 100.0]
# ])
# clusters = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

# centroids = find_centroid(coordinates, clusters)
# print(centroids)
