import argparse
import itertools
import math
import numpy as np
import gqlalchemy
from matplotlib import pyplot as plt

import kmeans

from collections import defaultdict
from typing import Iterator, Dict, List, Tuple, Any
from numpy.linalg import norm

NUMBER_OF_CLUSTERS = 4
TOP_N_SIMILARITIES = 5
OUTPUT_CHUNK_SIZE = 50

memgraph = gqlalchemy.Memgraph("127.0.0.1", 7687)


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Node2Vec Online Recommender',
    )
    subparsers = parser.add_subparsers(
        help='sub-command help',
        dest='action'
    )

    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Visualize k-means'
    )

    similarities_parser = subparsers.add_parser(
        'similarities',
        help='Add path of mage/dist to memgraph.conf '
    )
    similarities_parser.add_argument(
        '--n_clusters',
        help='Number of clusters',
        type=int,
        required=False
    )

    similarities_parser.add_argument(
        '--top_n_sim',
        help='Output top n similarities',
        type=int,
        required=False
    )

    return parser.parse_args()


class NodePairSimilarity:
    def __init__(self, node1: gqlalchemy.Node, node2: gqlalchemy.Node, similarity: float):
        self.node1: gqlalchemy.Node = node1
        self.node2: gqlalchemy.Node = node2
        self.similarity = similarity


def call_a_query(query: str) -> Iterator[Dict[str, Any]]:
    return memgraph.execute_and_fetch(query)


def get_node_embeddings() -> List[Tuple[gqlalchemy.Node, List[float]]]:
    rows = call_a_query("""CALL node2vec_online.get() YIELD node, embedding
                            RETURN node, embedding""")

    node_embeddings: List[Tuple[gqlalchemy.Node, List[float]]] = []
    for row in rows:
        node_embeddings.append((row['node'], row['embedding']))

    return node_embeddings


def get_labels(node_embeddings: List[Tuple[gqlalchemy.Node, List[float]]], number_of_clusters = NUMBER_OF_CLUSTERS) -> Dict[
    int, Tuple[gqlalchemy.Node, List[float]]]:
    nodes_embeddings_classes_dict = kmeans.get_groupings(number_of_clusters, node_embeddings)

    return nodes_embeddings_classes_dict


def get_cosine_similarity(node_embedding1: Tuple[gqlalchemy.Node, List[float]],
                          node_embedding2: Tuple[gqlalchemy.Node, List[float]]) -> float:
    node1, embedding1 = node_embedding1
    node2, embedding2 = node_embedding2
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    return math.fabs(np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2)))


def get_str_chunks(input: str, chunk_size) -> List[str]:
    return [input[i:i + chunk_size] for i in range(0, len(input), chunk_size)]


def form_chunk_output(input1: str, input2: str, chunk_size=OUTPUT_CHUNK_SIZE) -> List[List[str]]:
    input1_chunks = get_str_chunks(input1, chunk_size)
    input2_chunks = get_str_chunks(input2, chunk_size)
    formed_chunks = []
    for i in range(max(len(input1_chunks), len(input2_chunks))):
        desc1_out = " " * OUTPUT_CHUNK_SIZE if i >= len(input1_chunks) else input1_chunks[i]
        desc2_out = " " * OUTPUT_CHUNK_SIZE if i >= len(input2_chunks) else input2_chunks[i]
        formed_chunks.append([desc1_out, desc2_out])

    return formed_chunks


def get_node_pair_sim_output(node_pair_sim: NodePairSimilarity) -> List[List[str]]:
    output = []
    similarity = node_pair_sim.similarity

    description1: str = node_pair_sim.node1.properties.get('description')
    title1: str = node_pair_sim.node1.properties.get('title')
    id1: str = node_pair_sim.node1.properties.get('id')

    description2: str = node_pair_sim.node2.properties.get('description')
    title2: str = node_pair_sim.node2.properties.get('title')
    id2: str = node_pair_sim.node2.properties.get('id')

    output.append(["id: {id}".format(id=id1), "id: {id}".format(id=id2), "STATS"])
    output.append(["title: {title}".format(title=title1), "title: {title}".format(title=title2)])
    output.append(["", "", "similarity:{sim:.4f}".format(sim=similarity)])
    output.append(["\n"])

    output.extend(form_chunk_output(description1, description2))

    line_split = '-' * OUTPUT_CHUNK_SIZE
    output.append([line_split, line_split, line_split])
    output.append(["\n"])

    return output


def get_group_output(group_similarity: List[NodePairSimilarity], top_n_group_similarities=10) -> str:
    node_pair_sim_outputs = []
    for i in range(min(top_n_group_similarities, len(group_similarity))):
        node_pair_sim = group_similarity[i]
        node_pair_sim_output = get_node_pair_sim_output(node_pair_sim)
        node_pair_sim_outputs.extend(node_pair_sim_output)

    col_width = max(len(word) for row in node_pair_sim_outputs for word in row) + 2  # padding

    group_output = ""
    for row in node_pair_sim_outputs:
        group_output = group_output + "".join(word.center(col_width) for word in row) + "\n"
    return group_output


def calculate_similarities(classes_node_embeddings: Dict[int, Tuple[gqlalchemy.Node, List[float]]]) -> Dict[
    int, List[NodePairSimilarity]]:
    group_similarity_dict = defaultdict(list)

    for i in range(len(classes_node_embeddings)):
        group_similarity: List[NodePairSimilarity] = []
        for node_embedding1, node_embedding2 in itertools.combinations(classes_node_embeddings[i], 2):
            group_similarity.append(
                NodePairSimilarity(node_embedding1[0], node_embedding2[0],
                                   get_cosine_similarity(node_embedding1, node_embedding2)))
        group_similarity.sort(key=lambda x: x.similarity, reverse=True)
        group_similarity_dict[i] = group_similarity
    return group_similarity_dict


def visualize(embeddings: List[List[float]], clusters:List[int]):

    kmeans_inertia = kmeans.calculate_inertia(embeddings,clusters)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(2, 11), kmeans_inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


def main():
    arguments = get_arguments()
    if not hasattr(arguments, 'action') or arguments.action not in ['similarities', 'visualize']:
        print('''For usage run: python3 recommender -h ''')
        return

    node_embeddings = get_node_embeddings()

    if arguments.action == "visualize" :
        embeddings = [embedding for node, embedding in node_embeddings]
        visualize(embeddings, np.arange(2, 11))
        return

    if arguments.action == "similarities":
        top_n_sim=TOP_N_SIMILARITIES
        n_clusters = NUMBER_OF_CLUSTERS
        if arguments.n_clusters:
            n_clusters = int(arguments.n_clusters)
        if arguments.top_n_sim:
            top_n_sim=int(arguments.top_n_sim)
        node_embeddings_classes_dict = get_labels(node_embeddings, n_clusters)
        group_node_pair_similarity = calculate_similarities(node_embeddings_classes_dict)

        for i in range(len(group_node_pair_similarity)):
            print("GROUP: {group}".format(group=i))
            print(get_group_output(group_node_pair_similarity[i]),top_n_sim)

        return




if __name__ == "__main__":
    main()
