from Node import Node
import pickle
import os
import json
from tqdm import tqdm

#load data
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g1_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g1.pkl")
g2_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g1.pkl")
attributes_path = os.path.join(BASE_PATH, "2-Build_Graph/data/attributes.jsonl")

with open(g1_path, "rb") as f:
    nodes = pickle.load(f)

attribute_dict = {}
with open(attributes_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        node_id = line["entity_id"]
        attribute = line["summary"]
        attribute_dict[node_id] = attribute

#create A nodes
for node_id, attribute in tqdm(attribute_dict.items()):
    entity_node = nodes[node_id]
    if entity_node.node_type != "N" or not attribute.strip():
        raise ValueError(f"Summary for {node_id} is invalid")
    attribute_node_id = f"{node_id}:A000"
    attribute_node = Node(
        node_id = attribute_node_id,
        node_type = "A",
        content = attribute,
        source = node_id
    )
    attribute_node.link(entity_node)
    entity_node.link(attribute_node)
    nodes[attribute_node_id] = attribute_node

#save graph
with open(g2_path, "wb") as f:
    pickle.dump(nodes, f)

