import json
from Retrieval.ppr_local import shallow_ppr_local

entities_dict = {}
with open("2-Build_Graph/data/entities.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        entities_dict[line["entity"]] = line["nodes"]

def graph_retrieval(nodes, embedding_node_ids, question_entities):
    entity_node_ids = set()
    current_entities = {question_entities} if isinstance(question_entities, str) else set(question_entities)
    for e in current_entities:
        e = e.upper().strip()
        if e in entities_dict:
            entity_node_ids.update(entities_dict[e])

    for nid in embedding_node_ids: #from V nodes
        if nodes[nid].node_type == "V":
            for edge in nodes[nid].edges:
                if nodes[edge].node_type == "N":
                    entity_node_ids.add(edge)
    entry_node_ids = set(embedding_node_ids).union(entity_node_ids)

    ppr_search_results = shallow_ppr_local(nodes, entry_node_ids, ppr_context=None, debug=False)
    cross_node_ids = set(ppr_search_results.keys())
    all_nodes_ids = entry_node_ids.union(cross_node_ids)
    return [nid for nid in all_nodes_ids if nodes[nid].node_type not in {"N", "O"}]
