import json
import os
import pickle
import faiss
from Node import Node
#import numpy as np
import math

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g2_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g2.pkl")
g3_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g3.pkl")
communities_path = os.path.join(BASE_PATH, "2-Build_Graph/data/communities.jsonl")
faiss_path = os.path.join(DIR_PATH, "data", "embeddings.faiss")
embedding_processed_ids_path = f"{DIR_PATH}/data/embedding_processed_ids.txt"

#load data
with open(g2_path, "rb") as f:
    nodes = pickle.load(f)

communitiy_members = {}
community_summary = {}
community_overview = {}
with open(communities_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        communitiy_members[line["community_id"]] = set(line["members"])
        community_summary[line["community_id"]] = line["summary"]
        community_overview[line["community_id"]] = line["overview"]

with open(embedding_processed_ids_path, "r", encoding="utf-8") as f:
    embedding_processed_ids_list = [line.strip() for line in f if line.strip()]

index = faiss.read_index(faiss_path)
num_vectors = index.ntotal
dimension = index.d
#embeddings = np.zeros((num_vectors, dimension), dtype='float32')
#for i in range(num_vectors):
#    embeddings[i] = index.reconstruct(i)
embeddings = index.reconstruct_n(0, index.ntotal)

#get S,A,T,V,H embeddings
    #id lists
embedding_H_ids = []
embedding_SATV_ids = []
    #index
embedding_H_index = []
embedding_SATV_index = []


for i in range(num_vectors):
    current_id = embedding_processed_ids_list[i]
    id_components = list(current_id.split(":"))
    if len(id_components) == 1 and id_components[0][0] == "H":
        embedding_H_ids.append(current_id)
        embedding_H_index.append(i)
    else:
        embedding_SATV_ids.append(current_id)
        embedding_SATV_index.append(i)

H_embeddings = embeddings[embedding_H_index]
SATV_embeddings = embeddings[embedding_SATV_index]

#kmeans clustering
print("Begin clustering")
k = math.floor(math.sqrt(SATV_embeddings.shape[0]))
niter = 1000
kmeans = faiss.Kmeans(d=index.d,k=k,niter=niter,verbose=True,spherical=True,gpu=False)
kmeans.cp.metric_type = faiss.METRIC_INNER_PRODUCT
kmeans.train(SATV_embeddings)

centroids = kmeans.centroids
_, SATV_assignments = kmeans.index.search(SATV_embeddings, 1)
SATV_assignments = SATV_assignments.reshape(-1)

clusters = {i: set() for i in range(k)}
for idx, cid in enumerate(SATV_assignments):
    clusters[cid].add(embedding_SATV_ids[idx])

#assign H nodes
_, H_assignments = kmeans.index.search(H_embeddings, 3)
H_node_clusters = {}
for i in range(H_assignments.shape[0]):
    H_node_clusters[embedding_H_ids[i]] = set(H_assignments[i].tolist())

#create and link H and O nodes
for H_id in community_summary.keys():
    #ids
    O_id = f"{H_id}:O000"
    #data
    H_content = community_summary[H_id]
    O_content = community_overview[H_id]

    if not (isinstance(H_content, str) and isinstance(O_content, list)):
        print(H_content)
        print(O_content)
        raise ValueError(f"Invalid content at: {H_id}")
    #node creation
    H_node = Node(
        node_id = H_id,
        node_type = "H",
        source = "",
        content = H_content
    )

    O_node = Node(
        node_id = O_id,
        node_type = "O",
        source = H_id,
        content = O_content
    )

    #link H and O node
    H_node.link(O_node)
    O_node.link(H_node)

    #link H node with S,A,T,V nodes in the same community and cluster
    current_cluster = set()
    for cluster_id in H_node_clusters[H_id]:
        current_cluster |= clusters[cluster_id]
    current_community = communitiy_members[H_id]
    relevant_nodes = current_cluster & current_community

    for node_id in relevant_nodes:
        if node_id == H_id:
            continue
        relevant_node = nodes[node_id]
        relevant_node.link(H_node)
        H_node.link(relevant_node)
    
    #add to node list
    nodes[H_id] = H_node
    nodes[O_id] = O_node

#check
d_list = []
for node in nodes.values():
    if node.node_type == "H":
        d_list.append(node.getDegree())
print(min(d_list), max(d_list), sum(d_list)/ len(d_list), len(d_list))

with open(g3_path, "wb") as f:
    pickle.dump(nodes, f)