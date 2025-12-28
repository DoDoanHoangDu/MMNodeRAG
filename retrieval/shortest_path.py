import heapq
import time

def dijkstra_with_paths(graph, src):
    dist = {src: 0}
    if graph[src].node_type == "N" and isinstance(graph[src].source, set):
        for syn in graph[src].source:
            dist[syn] = 0
    prev = {}
    pq = [(value,key) for key,value in dist.items()]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u].edges.items():
            nd = d + 1/w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev

def reconstruct_path(prev, src, dst):
    path = []
    cur = dst

    while cur != src:
        if cur not in prev:
            return None  # no path
        path.append(cur)
        cur = prev[cur]

    path.append(src)
    path.reverse()
    return path

def all_pairs_shortest_paths(graph, entry_ids = None, debug = True):
    if debug:
        start = time.time()
    all_paths = {}

    for src in graph.keys():
        if entry_ids and src not in entry_ids:
            continue 
        _, prev = dijkstra_with_paths(graph, src)
        all_paths[src] = {}

        for dst in graph:
            if src == dst:
                all_paths[src][dst] = [src]
            else:
                all_paths[src][dst] = reconstruct_path(prev, src, dst)
    if debug:
        print(f"Reasoning time: {time.time() - start:.2f} seconds.")
    return all_paths
