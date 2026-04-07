from Node import Node
import pickle
import os
import json
from tqdm import tqdm
import igraph as ig
from leidenalg import find_partition, ModularityVertexPartition
from collections import defaultdict
from LLM.call_api import call_api
from LLM.prompts.high_level_elements_prompt import high_level_elements_prompt
from LLM.prompts.high_level_overview_prompt import high_level_overview_prompt


#load data
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g2_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g1.pkl")

with open(g2_path, "rb") as f:
    nodes = pickle.load(f)

#leiden commnunity detection
def leiden_community_detection(node_dict):
    id_to_idx = {nid: i for i, nid in enumerate(node_dict.keys())}
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    edge_weights = defaultdict(float)
    for nid, node in node_dict.items():
        for neighbor, weight in node.edges.items():
            a, b = id_to_idx[nid], id_to_idx[neighbor]
            if a < b:
                edge_weights[(a, b)] += weight

    edges = list(edge_weights.keys())
    weights = list(edge_weights.values())

    g = ig.Graph(n=len(node_dict), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = find_partition(g, ModularityVertexPartition, weights="weight")
    membership = partition.membership
    communities = {membership[i] : idx_to_id[i] for i in range(len(node_dict))}
    return communities

communities = leiden_community_detection(nodes)