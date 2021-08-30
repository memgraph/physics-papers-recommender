from collections import defaultdict
from typing import List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calculate_inertia(embeddings, clusters ):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    inertia = []

    for k in clusters:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings_scaled)
        print("k:", k, " inertia:", kmeans.inertia_)
        inertia.append(kmeans.inertia_)

    return inertia


def get_groups(number_of_clusters:int, node_embeddings: List[(any, List[float])])-> defaultdict[list]:
    scaler = StandardScaler()
    embeddings = [ embedding for node, embedding in node_embeddings]
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans:KMeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(embeddings_scaled)

    kmeans_labels = kmeans.labels_

    classes_dict = defaultdict(list)
    for i in range(len(kmeans_labels)):
        label = kmeans_labels[i]
        classes_dict[label].append([node_embeddings[0], node_embeddings[1]])


    return classes_dict