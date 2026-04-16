import os
import pickle
from Node import Node
import json
from collections import defaultdict

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
knn_path = os.path.join(BASE_PATH, "Evaluation/data/knn.jsonl")
entities_path = os.path.join(DIR_PATH, "data", "entities.jsonl")

#load data
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

entity_dict = defaultdict(list)
for node in nodes.values():
    if node.node_type == "N" or node.node_type == "O":
        current_ents = node.content
        if not isinstance(current_ents, set) and not isinstance(current_ents, list):
            raise ValueError(f"Content of {node.node_id} is not a list/set: {current_ents}")
        for ent in current_ents:
            entity_dict[ent].append(node.node_id)

with open(entities_path,"w",encoding="utf-8") as f:
    for ent in entity_dict:
        data = {
            "entity": ent,
            "nodes": entity_dict[ent]
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()