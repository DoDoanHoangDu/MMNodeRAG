import json

#main loop:
def sort_by_floats(strings, floats):
    paired = list(zip(strings, floats))
    paired.sort(key=lambda x: x[1], reverse=True)  # sort by float

    sorted_strings, sorted_floats = zip(*paired)
    return list(sorted_strings), list(sorted_floats)

def rerank_context(model, nodes, question, img_path, context_nodes):
    query = {"text": question, "image": img_path}
    context_nodes_content = []
    for c in context_nodes:
        if "V" in c:
            c_content = {"image": nodes[c].content}
        else:
            c_content = {"text": nodes[c].content}
        context_nodes_content.append(c_content)

    inputs = {
        "instruction": "Retrieve images or text relevant to the user's query.",
        "query": query,
        "documents": context_nodes_content,
    }
    scores = model.process(inputs)
    reranked_context_nodes, scores = sort_by_floats(context_nodes, scores)
    return [(reranked_context_nodes[i], scores[i]) for i in range(len(reranked_context_nodes))]