from LLM.prompts.question_decompose_prompt import question_decompose_prompt
from LLM.call_api import call_api
from Retrieval.retrieval import retrieve_relevant_nodes
import json
import numpy as np
import time
import torch

def get_context(query_context, graph_context, embedding_context, debug = True, reasoning = False):
    #Get values
    question = query_context['question']
    embedding_model = embedding_context['model']

    #Decompose question to extract entities
    start = time.time()
    prompt = question_decompose_prompt(question["text"])
    response_text, _ = call_api(prompt, model="", mode="self-host")
    finish_decomposition_time = time.time()
    if debug:
        print(f"Decomposition time: {finish_decomposition_time - start:.2f} seconds.")
    try:
        response_entities = json.loads(response_text)
    except Exception as e:
        print("Decomposed Question Response:", response_text)
        raise RuntimeError(f"JSON parse failed: {type(e).__name__}: {repr(e)}")
    #print('Response Parsed')
    if isinstance(response_entities, str):
        response_entities = [response_entities.upper()]
    else:
        response_entities = [ent.upper().strip() for ent in response_entities]
    if debug:
        print("Decomposed Question Response:", response_entities)
    #print("Response Processed")
    query_context['entities'] = response_entities
    if debug:
        print(query_context['entities'])
    #query_context['ppr']['k_ppr'] = len(query_context['entities'])
    
    #Get query embedding
    query_context['embedding'] = embedding_model.process(question).to(torch.float32).cpu().numpy()

    #Retrieve relevant nodes
    context = retrieve_relevant_nodes(graph_context, embedding_context, query_context, debug = debug, reasoning = reasoning)
    if debug:
        print(f"Retrieval time: {time.time() - finish_decomposition_time:.2f} seconds.")
    return context

