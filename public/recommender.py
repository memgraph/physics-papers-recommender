from typing import Iterator, Dict, Any, List

from gqlalchemy import Memgraph
import mgp
import kmeans

NUMBER_OF_GROUPS = 5

memgraph = Memgraph("127.0.0.1", 7687)


def call_a_query(query: str) -> Iterator[Dict[str, Any]]:
    return memgraph.execute_and_fetch(query)


def get_node_embeddings() -> List[(mgp.Vertex, List[float])]:
    rows = call_a_query("""CALL node2vec_online.get() YIELD node, embedding
                            RETURN node, embedding""")

    node_embeddings: List[(mgp.Vertex, List[float])] = []
    for row in rows:
        print("node", row['node'])
        print("embedding", row['embedding'])
        node_embeddings.append((row['node'], row['embedding']))

    return node_embeddings


def get_labels(node_embeddings: List[(mgp.Vertex, List[float])]):
    nodes_embeddings_classes_dict = kmeans.get_groups(NUMBER_OF_GROUPS, node_embeddings)

    return nodes_embeddings_classes_dict


def main():
    node_embeddings = get_node_embeddings()
    node_embeddings_classes_dict = get_labels(node_embeddings)
    print(node_embeddings_classes_dict[0])


if __name__ == "__main__":
    main()
