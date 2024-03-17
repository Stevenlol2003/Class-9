import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    # Assumes the file exists and is a properly formatted CSV
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
            
    return data

def calc_features(row):
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    
    features = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)

    return features

#def hac(features):
#    complete_linkage = linkage(features, method='complete', metric='euclidean')
#    
#    return complete_linkage

# gave up on this, could not fully debug it
def hac(features):
    n = len(features)  # Number of data points

    # Step 1: Initialize cluster numbers
    cluster_numbers = np.arange(n)

    # Step 2: Create the linkage matrix
    Z = np.zeros((n - 1, 4))  # Initialize the linkage matrix
    linkage_index = n  # Index to track newly created clusters

    # Precompute distances
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(features[i] - features[j])  # Euclidean distance
            distances[i, j] = distance
            distances[j, i] = distance

    for i in range(n - 1):
        # (a) Find the two closest clusters
        min_distance = np.inf
        min_distance_index = None
        for j in range(n):
            for k in range(j + 1, n):
                distance = distances[j, k]
                if distance < min_distance:
                    min_distance = distance
                    min_distance_index = (j, k)
        
        # Check if a valid min_distance_index was found
        if min_distance_index is not None:
            cluster1, cluster2 = min_distance_index

            # (b) Put cluster numbers and distance into Z
            Z[i, 0] = cluster1
            Z[i, 1] = cluster2
            Z[i, 2] = min_distance

            # (c) Calculate the size of the new cluster
            new_cluster_size = (
                1 if cluster_numbers[cluster1] != cluster_numbers[cluster2] else
                np.count_nonzero(cluster_numbers == cluster1)
            )
            Z[i, 3] = new_cluster_size

            # Update cluster numbers to reflect the merge
            cluster_numbers[cluster_numbers == cluster1] = linkage_index
            cluster_numbers[cluster_numbers == cluster2] = linkage_index

            # Update the distances matrix with the new cluster
            distances = np.insert(distances, linkage_index, np.inf, axis=0)
            distances = np.insert(distances, linkage_index, np.inf, axis=1)

            for j in range(linkage_index):
                if j != linkage_index:
                    new_distance = min(distances[cluster1, j], distances[cluster2, j])
                    distances[linkage_index, j] = new_distance
                    distances[j, linkage_index] = new_distance

            # Set the distances of the merged clusters to infinity to avoid re-merging
            distances[cluster1, :] = np.inf
            distances[:, cluster1] = np.inf
            distances[cluster2, :] = np.inf
            distances[:, cluster2] = np.inf

            linkage_index += 1

    # Step 3: Convert Z into a NumPy array if it isn't already
    if not isinstance(Z, np.ndarray):
        Z = np.array(Z)

    return Z

def fig_hac(Z, names):
    fig = plt.figure()
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    return fig

def normalize_features(features):
    mean = np.mean(features, axis=0)
    sd = np.std(features, axis=0)
    normalized_features = []

    for feature in features:
        normalized_feature = (feature - mean) / sd
        normalized_features.append(normalized_feature)

    return normalized_features

if __name__ == "__main__":
    data = load_data('countries.csv')
    # print(data)
    country_names = [row['Country'] for row in data]
    # print(country_names)
    features = [calc_features(row) for row in data]
    # print("Features -------------------------")
    # print(features)
    features_normalized = normalize_features(features)
    n = 10
    # print(features_normalized)
    Z_raw = hac(features[:n])
    #print("Z_raw -------------------------")
    #print(Z_raw)
    Z_normalized = hac(features_normalized[:n])
    print(Z_normalized)
    fig = fig_hac(Z_normalized, country_names[:n])
    plt.show()
    #complete_clustering = linkage(features, method="complete", metric="euclidean")
    #dendrogram(complete_clustering)
    #plt.show()