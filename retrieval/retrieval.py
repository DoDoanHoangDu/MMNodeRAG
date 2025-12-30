import re
from graphs.Node import Node
from retrieval.ppr_local import shallow_ppr_local
from retrieval.shortest_path import all_pairs_shortest_paths


def find_relevant_embeddings(embedding_index, query_embedding, k_embedding):
    similarity, idx = embedding_index.search(query_embedding, k_embedding)
    return similarity[0], idx[0]

def find_relevant_entities(graph_context,query_entities):
    graph_entities = graph_context['entities']
    graph_overview = graph_context['overview']
    if isinstance(query_entities, str):
        query_set = {query_entities}
    else:
        query_set = set(query_entities) 
    entity_ids = set()
    for e in query_set:
        e = e.upper()
        if e in graph_entities:
            entity_ids.add(graph_entities[e])
        mask = graph_overview['community_overview'].str.contains(fr"\b{re.escape(e)}\b", case=False, na=False, regex=True)

        matching_node_ids = graph_overview.loc[mask, 'node_id'].tolist()
        entity_ids.update(matching_node_ids)
    return entity_ids

def retrieve_relevant_nodes(graph_context, embedding_context, query_context, debug = True, reasoning = False):
    #Get values:
    query_entities = query_context['entities']
    query_embedding = query_context['embedding']
    k_embedding = query_context['k_embedding']
    ppr_context = query_context['ppr']

    embedding_index = embedding_context['index']
    embedding_ids = embedding_context['ids']

    graph = graph_context['graph']

    #find relevant embeddings
    sim, idx = find_relevant_embeddings(embedding_index, query_embedding, k_embedding)
    embedding_node_ids = [embedding_ids[i] for i in idx]

    #find relevant entities
    entity_node_ids = find_relevant_entities(graph_context,query_entities)

    #combine entry node ids
    entry_node_ids = set(embedding_node_ids).union(entity_node_ids)
    if debug:
        print("Number of entry nodes:", len(entry_node_ids))

    #perform PPR from entry nodes to find cross nodes
    ppr_search_results = shallow_ppr_local(graph, entry_node_ids, ppr_context, debug = debug)
    cross_node_ids = set(ppr_search_results.keys())
    all_nodes_ids = entry_node_ids.union(cross_node_ids)
    if reasoning:
        #find shortest paths between entry nodes
        reasoning_node_ids = set()
        reasoning_entities = [node_id for node_id in entry_node_ids if graph[node_id].node_type in ['N']]
        paths = all_pairs_shortest_paths(graph, reasoning_entities, debug = debug)
        for i in reasoning_entities:
            for j in reasoning_entities:
                path_ij = paths[i][j]
                if not path_ij or len(path_ij) <= 2:
                    continue
                for node_id in path_ij:
                    if node_id not in all_nodes_ids:
                        #print("Adding reasoning node:", node_id)
                        all_nodes_ids.add(node_id)
                        reasoning_node_ids.add(node_id)
        if debug:
            print("Number of reasoning nodes added:", len(reasoning_node_ids))
            #print("Reasoning nodes:", reasoning_node_ids)
    #content
    content = {}
    for node_id in all_nodes_ids:
        node = graph[node_id]
        if node.node_type in ['N','O']: #remove non-informative nodes
            continue
        content[node_id] = node.content
    return content