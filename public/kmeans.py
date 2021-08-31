from collections import defaultdict
from typing import List, Tuple, Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calculate_inertia(embeddings:List[List[float]], clusters:List[int])->List[float]:
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    inertia = []

    for k in clusters:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings_scaled)
        inertia.append(kmeans.inertia_)

    return inertia

def get_groupings(number_of_clusters: int, node_embeddings: List[Tuple[any, List[float]]]) -> Dict[Any, list]:
    scaler = StandardScaler()
    embeddings = [embedding for node, embedding in node_embeddings]
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans: KMeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(embeddings_scaled)
    kmeans_labels = kmeans.labels_

    classes_dict = defaultdict(list)
    for i in range(len(kmeans_labels)):
        label = kmeans_labels[i]
        classes_dict[label].append((node_embeddings[i][0], embeddings_scaled[i]))

    return classes_dict
