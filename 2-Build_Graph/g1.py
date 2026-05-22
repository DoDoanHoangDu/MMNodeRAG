import os
import networkx as nx
import json
from tqdm import tqdm
import re
from Node import Node
import pickle

#file paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
chunks_path =os.path.join(BASE_PATH, "1-Preprocess/data/chunks.jsonl")
decomposition_path = os.path.join(BASE_PATH, "1-Preprocess/data/decomposition.jsonl")
image_mapping_path = os.path.join(BASE_PATH, "1-Preprocess/data/image_entity_mapping.jsonl")
synonym_graph_path = os.path.join(BASE_PATH, "1-Preprocess/data/synonym_graph.edgelist")
save_path = os.path.join(DIR_PATH, "data/g1.pkl")
print(save_path)

#load synonym graph and create synonym dictionary
synonym_graph = nx.read_edgelist(synonym_graph_path, delimiter = "\t", create_using=nx.Graph, nodetype=str)
node_names = synonym_graph.nodes()
for node_name in node_names:
    if not isinstance(node_name, str):
        raise ValueError("Invalid synonym graph data")
 
synonym_groups = list(nx.connected_components(synonym_graph))
synonym_dict = {}
max_group_size = 0
for group in synonym_groups:
    max_group_size = max(max_group_size, len(group))
    for e in group:
        synonym_dict[e] = group

#load image entity mapping
image_entity_mapping = {}
with open(image_mapping_path, "r", encoding = "utf-8") as f:
    for line in tqdm(f, desc = "Loading image entity mapping"):
        line = json.loads(line)
        key = line["image_file"]
        value = line["entities"]
        if not isinstance(key, str):
            raise ValueError("Invalid image mapping data")
        if not isinstance(value, list):
            raise ValueError("Invalid image mapping data")
        for e in value:
            if not isinstance(e, str):
                raise ValueError("Invalid image mapping data")
        image_entity_mapping[key] = value

#function to match entities with relationship
def entities_in_relationship(rel_node, ent_nodes):
    rel_text = rel_node.content.upper()
    found = set()
    for e_node in ent_nodes:
        matched = False
        for e in e_node.content:
            pattern = r"\b" + re.escape(e.strip().upper()) + r"\b"
            if re.search(pattern, rel_text):
                matched = True
                break
        if matched: found.add(e_node)
    return found

#create T nodes and link them sequentially
nodes = dict()
with open(chunks_path, "r", encoding = "utf-8") as f:
    for line in tqdm(f, desc = "Creating T nodes"):
        line = json.loads(line)
        chunk_id = line["chunk_id"]
        chunk_content = line["chunk_content"]
        chunk_node = Node(
            node_id = chunk_id,
            node_type = "T",
            source = chunk_id.split(":")[0],
            content = chunk_content
        )
        nodes[chunk_id] = chunk_node
for chunk_id in nodes:
    id_components = chunk_id.split(":")
    chunk_index = int(id_components[1][1:])
    if chunk_index > 0:
        prev_chunk_id = f"{id_components[0]}:T{chunk_index-1:03d}"
        if prev_chunk_id in nodes:
            nodes[chunk_id].link(nodes[prev_chunk_id])
    next_chunk_id = f"{id_components[0]}:T{chunk_index+1:03d}"
    if next_chunk_id in nodes:
        nodes[chunk_id].link(nodes[next_chunk_id])

#create and link S, N, R nodes
entity_nodes = dict()
entity_counter = 0
with open(decomposition_path, "r", encoding = "utf-8") as f:
    for line in tqdm(f, desc = "Creating nodes"):
        line = json.loads(line)
        chunk_id = line["chunk_id"]
        response = line["response"]
        for unit_idx, unit in enumerate(response):
            #data
            semantic_unit = unit["semantic_unit"]
            entities = unit["entities"]
            relationships = unit["relationships"]
 
            #ids
            semantic_id = f"{chunk_id}:S{unit_idx:03d}"
            relationship_ids = [f"{semantic_id}:R{r_idx:03d}" for r_idx in range(len(relationships))]
 
            #create semantic node
            semantic_node = Node(
                node_id = semantic_id,
                node_type = "S",
                source = chunk_id,
                content = semantic_unit
            )
            nodes[semantic_id] = semantic_node
 
            #create entity nodes
            current_entity_nodes = set()
            for e_idx, entity in enumerate(entities):
                key = entity.strip().upper()
                synonyms = synonym_dict.get(key, {key})
                if key not in entity_nodes.keys():
                    entity_id = f"N{entity_counter:06d}"
                    entity_counter += 1
                    entity_node = Node(
                        node_id = entity_id,
                        node_type = "N",
                        source = None,
                        content = synonyms
                    )
                    nodes[entity_id] = entity_node
                    for syn in synonyms:
                        entity_nodes[syn.upper().strip()] = entity_node
                else:
                    entity_node = entity_nodes[key]
                current_entity_nodes.add(entity_node)
 
            #create relationship nodes
            current_relationship_nodes = set()
            for r_idx, relationship in enumerate(relationships):
                relationship_node = Node(
                    node_id = relationship_ids[r_idx],
                    node_type = "R",
                    source = semantic_id,
                    content = relationship
                )
                current_relationship_nodes.add(relationship_node)
                nodes[relationship_ids[r_idx]] = relationship_node
 
            #link nodes
            chunk_node = nodes[chunk_id]
            chunk_node.link(semantic_node)
            semantic_node.link(chunk_node)
            for entity_node in current_entity_nodes:
                semantic_node.link(entity_node)
                entity_node.link(semantic_node)
 
            for relationship_node in current_relationship_nodes:
                ents = entities_in_relationship(relationship_node, current_entity_nodes)
                for ent in ents:
                    relationship_node.link(ent)
                    ent.link(relationship_node)
    entities_dict = {k:v.node_id for k,v in entity_nodes.items()}


#create visual node and link to entities
for image_file, matched_entities in image_entity_mapping.items():
    image_path = f"Dataset/wiki_images_resized/{image_file}"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    image_id = f"{image_file.split('.')[0]}:V000"
    visual_node = Node(
        node_id = image_id,
        node_type = "V",
        source = image_file.split('.')[0],
        content = image_path
    )
    for e in matched_entities:
        entity_node = entity_nodes[e]
        entity_node.link(visual_node)
        visual_node.link(entity_node)
    nodes[image_id] = visual_node

#handle disconnected relationship nodes
disconnected_nodes = set()
for node in nodes.values():
    if not node.edges:
        disconnected_nodes.add(node)
for node in tqdm(disconnected_nodes, desc="Linking disconnected nodes"):
    if node.node_type != "R":
        raise ValueError("Only relationship nodes should be disconnected at this stage")
    ents = entities_in_relationship(node, entity_nodes.values())
    for ent in ents:        
        node.link(ent)
        ent.link(node)

#print statistics
avg_degree = 0
min_edge, max_edge = float("inf"), 0
min_degree, max_degree = float("inf"), 0
node_statistics = {}
for node in nodes.values():
    #node.print()
    min_degree = min(min_degree, node.degree)
    max_degree = max(max_degree, node.degree)
    avg_degree += node.degree
    node_statistics[node.node_type] = node_statistics.get(node.node_type, 0) + 1
    edges = node.edges.values()
    if edges:
        min_edge = min(min_edge, min(edges))
        max_edge = max(max_edge, max(edges))

avg_degree /= len(nodes)

disconnected_nodes = set()
for node in nodes.values():
    if not node.edges:
        disconnected_nodes.add(node)

print(f"Number of disconnected nodes: {len(disconnected_nodes)}")
print(f"Min edge: {min_edge}")
print(f"Max edge: {max_edge}")
print(f"Min degree: {min_degree}")
print(f"Avg degree: {avg_degree:.2f}")
print(f"Max degree: {max_degree}")
print("Node type distribution:")
for node_type, count in node_statistics.items():
    print(f"{node_type}: {count}")

#save graph
with open(save_path, "wb") as f:
    pickle.dump(nodes, f)