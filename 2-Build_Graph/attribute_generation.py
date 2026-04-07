from Node import Node
from LLM.call_api import call_api
from LLM.prompts.attribute_generation_prompt import attribute_generation_prompt
import pickle
import math
import os
import re
import json
import time
from tqdm import tqdm
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g1_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g1.pkl")
with open(g1_path, "rb") as f:
    nodes = pickle.load(f)

#find k-core important nodes: entity nodes with number of connection >= k_default
def k_default(nodes):
    V = len(nodes)
    sum_deg = sum([node.degree for node in nodes.values()])
    k_avg = sum_deg / V
    k_def = math.floor(math.log(V) * math.sqrt(k_avg))
    return k_def

def k_core_importance(nodes):
    k = k_default(nodes)
    important_nodes = {node_id: node for node_id, node in nodes.items() if node.degree >= k and node.node_type == "N"}
    return important_nodes

k_core_nodes = k_core_importance(nodes)
print(f"Number of k-core important nodes: {len(k_core_nodes)}")

#sort and collect neighbors of each important node
def sort_key(id_str):
    return list(map(int, re.findall(r'\d+', id_str)))

def get_semantic_units_and_relationships(node_id, all_nodes):
    semantic_units = []
    relationships = []
    node = all_nodes[node_id]
    for related_node_id in node.edges.keys():
        related_node = all_nodes[related_node_id]
        if related_node.node_type == "S":
            semantic_units.append(related_node.node_id)
        elif related_node.node_type == "R":
            relationships.append(related_node.node_id)
        else:
            continue
    return [sorted(semantic_units, key=sort_key), sorted(relationships, key=sort_key)]

important_nodes_neighbors = {node_id: get_semantic_units_and_relationships(node_id, nodes) for node_id in k_core_nodes}

#join neighboring content of each important nodes to create context
def format_list(l):
    ans = []
    for i in range(len(l)):
        ans.append(f"[{i+1}] {l[i]}")
    return "\n".join(ans)

def get_context(neighbors_dict, all_nodes):
    context_dict = {}
    for node_id, neighbors in neighbors_dict.items():
        semantic_context_pieces = []
        relationship_context_pieces = []
        for neighbor in neighbors[0]:
            neighbor_node = all_nodes[neighbor]
            neighbor_content = neighbor_node.content
            semantic_context_pieces.append(neighbor_content)
        for neighbor in neighbors[1]:
            neighbor_node = all_nodes[neighbor]
            neighbor_content = neighbor_node.content
            relationship_context_pieces.append(neighbor_content)
        context_dict[node_id] = [format_list(semantic_context_pieces), format_list(relationship_context_pieces)]
    return context_dict

important_nodes_context = get_context(important_nodes_neighbors, nodes)

lengths = [len(v[0].split()) + len(v[1].split()) for v in important_nodes_context.values()]
print(f"Min context length: {min(lengths)}")
print(f"Max context length: {max(lengths)}")
print(f"Average context length: {sum(lengths) / len(lengths)}")


#save context
output_path = os.path.join(BASE_PATH, "2-Build_Graph/data/attributes.jsonl")
processed_ids = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            node_id = line["entity_id"]
            node = nodes[node_id]
            if node.node_type != "N":
                raise ValueError("Invalid entity id saved")
            else:
                processed_ids.add(node_id)
print(f"Processed entities: {len(processed_ids)}")

with open(output_path, "a", encoding="utf-8") as f:
    for node_id,context in tqdm(important_nodes_context.items()):
        if node_id in processed_ids:
            continue
        node = nodes[node_id]
        if node.node_type != "N":
            raise ValueError("Invalid entity id in important nodes")
        entities = nodes[node_id].content #content = a list of synonyms
        semantic_units = context[0]
        relationships = context[1]
        prompt = attribute_generation_prompt(entities, semantic_units, relationships)
        MAX_ATTEMPTS = 30
        for attempt in range(1, MAX_ATTEMPTS+1):
            try:
                response, token = call_api(prompt)
                line = {
                    "entity_id": node_id,
                    "semantic_units": semantic_units,
                    "relationships": relationships,
                    "summary": response,
                    "token": token
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
                break
            except Exception as e:
                if attempt == MAX_ATTEMPTS:
                    print(f"Failed on entity {node_id} with context length {len(prompt.split())}: {e}")
                    continue
                time.sleep(5 * attempt)

print("Finished")
            

