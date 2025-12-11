from answering.question_decompose_prompt import question_decompose_prompt
from retrieval.retrieval import retrieve_relevant_nodes
from google import genai
import os
import json
import numpy as np
import time

def call_gemini(text, API_KEY):
    client = genai.Client(api_key = API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config={ "response_mime_type": "application/json"}
    )
    return response.text

def get_context(query_context, graph_context, embedding_context, API_KEY):
    #Get values
    question = query_context['question']
    embedding_model = embedding_context['model']

    graph = graph_context['graph']
    graph_entities = graph_context['entities']

    #Decompose question to extract entities
    start = time.time()
    prompt = question_decompose_prompt(question)
    response_text = call_gemini(prompt, API_KEY)
    print("Decomposed Question Response:", response_text)
    finish_decomposition_time = time.time()
    print(f"Decomposition time: {finish_decomposition_time - start:.2f} seconds.")
    try:
        response_entities = json.loads(response_text)
    except Exception as e:
        raise RuntimeError(f"JSON parse failed: {type(e).__name__}: {repr(e)}")
    print('Response Parsed')
    if isinstance(response_entities, str):
        response_entities = [response_entities.upper()]
    else:
        response_entities = [ent.upper() for ent in response_entities]
    print("Response Processed")
    query_context['entities'] = response_entities
    #Extend to synonyms
    all_synonyms = []
    for ent in response_entities:
        if ent not in graph_entities:
            continue
        print("Processing entity:",ent)
        entity_node_id = graph_entities[ent]
        entity_node = graph[entity_node_id]
        if not entity_node.source:
            continue
        synonym_ids = list(entity_node.source.split(","))
        synonyms = [graph[synonym_id].content for synonym_id in synonym_ids]
        all_synonyms.extend(synonyms)
    print("Synonyms found:", len(all_synonyms))
    query_context['entities'].extend(all_synonyms)
    print(query_context['entities'])
    
    #Get query embedding
    query_context['embedding'] = embedding_model.encode([question], convert_to_numpy=True).astype(np.float32)

    #Retrieve relevant nodes
    context = retrieve_relevant_nodes(graph_context, embedding_context, query_context)
    print(f"Retrieval time: {time.time() - finish_decomposition_time:.2f} seconds.")
    return context

