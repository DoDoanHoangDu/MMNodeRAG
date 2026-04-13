import os
import pickle
import faiss
from Node import Node
import time

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g3_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g3.pkl")
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
faiss_path = os.path.join(DIR_PATH, "data", "embeddings.faiss")
hnsw_path = f"{DIR_PATH}/data/embeddings_hnsw.faiss"
embedding_processed_ids_path = f"{DIR_PATH}/data/embedding_processed_ids.txt"

#load data
with open(g3_path, "rb") as f:
    nodes = pickle.load(f)

with open(embedding_processed_ids_path, "r", encoding="utf-8") as f:
    embedding_processed_ids_list = [line.strip() for line in f if line.strip()]

index = faiss.read_index(faiss_path)
num_vectors = index.ntotal
dimension = index.d
embeddings = index.reconstruct_n(0, index.ntotal) #already normalized

#sanity check
sum_edge = 0
for node_id in nodes:
    node = nodes[node_id]
    for edge in node.edges:
        if node.edges[edge] > 1 or edge == node_id:
            raise ValueError(f"Invalid edge from {node_id} to {edge}: {node.edges[edge]}")
        sum_edge += node.edges[edge]
print("Total edge weight:", sum_edge/2)

#build hnsw index
start_time = time.time()
M = 32
hnsw = faiss.IndexHNSWFlat(embeddings.shape[1], M, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 200
hnsw.hnsw.efSearch = 64
hnsw.add(embeddings)
print(f"HNSW index built in {time.time() - start_time:.2f} seconds.")

#add hnsw edges to graph
start_time = time.time()
added_semantic_edges = set()
k = 16
similarities, indices = hnsw.search(embeddings, k+1)
for i in range(len(embedding_processed_ids_list)):
    node_id = embedding_processed_ids_list[i]
    node = nodes[node_id]
    for j in range(k+1):
        if i not in indices[indices[i][j]]:
            continue
        neighbor_id = embedding_processed_ids_list[indices[i][j]]
        if neighbor_id == node_id:
            continue
        neighbor = nodes[neighbor_id]
        similarity = float(similarities[i][j])
        if similarity <= 0.5:
            continue

        edge_key = frozenset((node_id, neighbor_id))
        if edge_key in added_semantic_edges:
            continue
        added_semantic_edges.add(edge_key)

        node.link(neighbor, similarity)
        neighbor.link(node, similarity)
print(f"HNSW edge linking in {time.time() - start_time:.2f} seconds.")

#calculate total edge weight after adding hnsw edges
sum_edge = 0
for node_id in nodes:
    node = nodes[node_id]
    for edge in node.edges:
        if edge == node_id:
            raise ValueError(f"Invalid edge from {node_id} to {edge}: {node.edges[edge]}")
        sum_edge += node.edges[edge]
print("Total edge weight:", sum_edge/2)

#save
faiss.write_index(hnsw, hnsw_path)
with open(g4_path, "wb") as f:
    pickle.dump(nodes, f)