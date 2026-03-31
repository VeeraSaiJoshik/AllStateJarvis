"""
NAME: Dijkstra's Shortest Path Algorithm
TAGS: graph, shortest-path, greedy, heap, weighted, sssp
DESCRIPTION: Finds single-source shortest paths in graphs with non-negative edge weights.
    Use whenever edge weights are non-negative; it is faster than Bellman-Ford for sparse
    graphs. Returns distance array and supports path reconstruction via parent tracking.
COMPLEXITY: Time O((V + E) log V), Space O(V + E)
"""

import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# ──────────────────────────────────────────────
# Graph helper: adjacency list with weights
# edges: list of (u, v, w)
# ──────────────────────────────────────────────

def build_graph(n: int, edges: List[Tuple[int, int, int]],
                directed: bool = False) -> defaultdict:
    g: defaultdict = defaultdict(list)
    for u, v, w in edges:
        g[u].append((v, w))
        if not directed:
            g[v].append((u, w))
    return g


# ──────────────────────────────────────────────
# 1. Standard Dijkstra — distance array only
# ──────────────────────────────────────────────

def dijkstra(graph: defaultdict, src: int, n: int) -> List[float]:
    """
    Returns dist[] where dist[v] = shortest distance from src to v.
    Uses a min-heap (lazy deletion). Nodes 0-indexed, n total.
    dist[v] = inf if v is unreachable.
    """
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0
    heap = [(0, src)]          # (distance, node)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:        # stale entry — skip
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist


# ──────────────────────────────────────────────
# 2. Dijkstra with parent tracking (path reconstruction)
# ──────────────────────────────────────────────

def dijkstra_with_path(graph: defaultdict, src: int, n: int
                       ) -> Tuple[List[float], List[int]]:
    """
    Returns (dist, parent).
    parent[v] = predecessor of v on the shortest path from src (-1 if none/src).
    """
    INF = float('inf')
    dist = [INF] * n
    parent = [-1] * n
    dist[src] = 0
    heap = [(0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, parent


def reconstruct_path(parent: List[int], src: int, dst: int) -> Optional[List[int]]:
    """Reconstruct shortest path from src to dst using parent array."""
    if parent[dst] == -1 and dst != src:
        return None
    path = []
    node = dst
    while node != -1:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path if path[0] == src else None


# ──────────────────────────────────────────────
# 3. Dijkstra — all-pairs (run from each vertex)
#    Only practical for small V or sparse graphs.
# ──────────────────────────────────────────────

def dijkstra_all_pairs(graph: defaultdict, n: int) -> List[List[float]]:
    """Returns dist[s][v] = shortest path from s to v for all s."""
    return [dijkstra(graph, s, n) for s in range(n)]


# ──────────────────────────────────────────────
# 4. Dijkstra on edge list (builds graph internally)
#    Handy for competition input parsing.
# ──────────────────────────────────────────────

def sssp(n: int, edges: List[Tuple[int, int, int]], src: int,
         directed: bool = True) -> List[float]:
    """
    Single-source shortest path from src.
    edges: list of (u, v, weight).  Nodes 0-indexed.
    """
    graph = build_graph(n, edges, directed)
    return dijkstra(graph, src, n)


# ──────────────────────────────────────────────
# 5. K-th shortest path / Dijkstra variant for
#    problems requiring the k-th arrival at dest.
#    Uses a counter per node; pops at most k times per node.
# ──────────────────────────────────────────────

def dijkstra_kth_shortest(graph: defaultdict, src: int, dst: int,
                           k: int, n: int) -> float:
    """
    Returns the k-th shortest path distance from src to dst.
    Returns inf if fewer than k paths exist.
    Note: k=1 is the standard shortest path.
    """
    INF = float('inf')
    count = [0] * n
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        count[u] += 1
        if u == dst and count[u] == k:
            return d
        if count[u] > k:
            continue
        for v, w in graph[u]:
            heapq.heappush(heap, (d + w, v))
    return INF


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Graph: 0->1 (4), 0->2 (1), 2->1 (2), 1->3 (1), 2->3 (5)
    edges = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
    g = build_graph(4, edges, directed=True)

    dist = dijkstra(g, 0, 4)
    print("Distances from 0:", dist)           # [0, 3, 1, 4]

    dist, par = dijkstra_with_path(g, 0, 4)
    print("Path 0->3:", reconstruct_path(par, 0, 3))   # [0, 2, 1, 3]

    print("All-pairs:", dijkstra_all_pairs(g, 4))

    print("SSSP (edge list):", sssp(4, edges, 0, directed=True))

    print("2nd shortest path 0->3:", dijkstra_kth_shortest(g, 0, 3, 2, 4))
