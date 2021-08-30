import itertools
import math
from collections import defaultdict
from operator import itemgetter
from typing import Iterator, Dict, Any, List, Tuple
from numpy.linalg import norm

import numpy as np
from gqlalchemy import Memgraph
# import mgp
import kmeans

NUMBER_OF_GROUPS = 4

memgraph = Memgraph("127.0.0.1", 7687)


def call_a_query(query: str) -> Iterator[Dict[str, Any]]:
    return memgraph.execute_and_fetch(query)


def get_node_embeddings() -> List[Tuple[any, List[float]]]:
    rows = call_a_query("""CALL node2vec_online.get() YIELD node, embedding
                            RETURN node, embedding""")

    node_embeddings: List[Tuple[any, List[float]]] = []
    for row in rows:
        node_embeddings.append((row['node'], row['embedding']))

    return node_embeddings

#defaultdict[int, Tuple[Any, List[float]]]
def get_labels(node_embeddings: List[Tuple[any, List[float]]]):
    nodes_embeddings_classes_dict = kmeans.get_groups(NUMBER_OF_GROUPS, node_embeddings)

    return nodes_embeddings_classes_dict


def get_cosine_similarity(node_embedding1: Tuple[any, List[float]], node_embedding2: Tuple[any, List[float]]) -> float:
    node1, embedding1 = node_embedding1
    node2, embedding2 = node_embedding2
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    return math.fabs(np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2)))

#defaultdict[int, Tuple[Any, List[float]]]
def calculate_similarities(classes_node_embeddings, top_n_group_similarities=10):
    for i in range(len(classes_node_embeddings)):
        group_similarity = []
        for node_embedding1, node_embedding2 in itertools.combinations(classes_node_embeddings[i], 2):
            group_similarity.append([get_cosine_similarity(node_embedding1, node_embedding2), node_embedding1[0], node_embedding2[0]])

        #print("group:", i, " ,mean:", np.mean(np.array([x[0] for x in group_similarity])))
        group_similarity = sorted(group_similarity, key=itemgetter(0))
        list.reverse(group_similarity)
        print("group", i)
        for i in range(top_n_group_similarities):
            similarity, node1, node2 = group_similarity[i]
            print(similarity,"\n\t", node1,"\n\t", node2, "\n")


def main():
    node_embeddings = get_node_embeddings()
    node_embeddings_classes_dict = get_labels(node_embeddings)
    calculate_similarities(node_embeddings_classes_dict)


if __name__ == "__main__":
    main()
