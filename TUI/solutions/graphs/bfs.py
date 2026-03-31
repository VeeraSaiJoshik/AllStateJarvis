"""
NAME: Breadth-First Search (BFS)
TAGS: graph, bfs, shortest-path, unweighted, multi-source
DESCRIPTION: Explores a graph level-by-level, guaranteeing shortest paths in unweighted
    graphs. Use for shortest path in unit-weight graphs, level-order traversal, and
    multi-source BFS to find distances from multiple starting nodes simultaneously.
COMPLEXITY: Time O(V + E), Space O(V)
"""

from collections import deque, defaultdict
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────
# Graph helper
# ──────────────────────────────────────────────

def build_graph(n: int, edges: List[Tuple[int, int]], directed: bool = False) -> defaultdict:
    g: defaultdict = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
        if not directed:
            g[v].append(u)
    return g


# ──────────────────────────────────────────────
# 1. Basic BFS — returns visit order
# ──────────────────────────────────────────────

def bfs(graph: defaultdict, start: int) -> List[int]:
    """BFS traversal from `start`. Returns nodes in visit order."""
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return order


# ──────────────────────────────────────────────
# 2. BFS Shortest Path (unweighted)
#    Returns distance array and parent array for path reconstruction.
# ──────────────────────────────────────────────

def bfs_shortest_path(graph: defaultdict, src: int, n: int) -> Tuple[List[int], List[int]]:
    """
    Single-source shortest path on an unweighted graph.
    Returns (dist, parent) where dist[v] = shortest distance from src to v (-1 if unreachable),
    and parent[v] = predecessor of v on shortest path (-1 if none).
    Nodes assumed 0-indexed, n total nodes.
    """
    INF = -1
    dist = [INF] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for nb in graph[node]:
            if dist[nb] == INF:
                dist[nb] = dist[node] + 1
                parent[nb] = node
                queue.append(nb)
    return dist, parent


def reconstruct_path(parent: List[int], src: int, dst: int) -> Optional[List[int]]:
    """Reconstruct path from src to dst using the parent array."""
    if parent[dst] == -1 and dst != src:
        return None          # unreachable
    path = []
    node = dst
    while node != -1:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path if path[0] == src else None


# ──────────────────────────────────────────────
# 3. Multi-Source BFS
#    Starts BFS simultaneously from all sources.
#    Common use: distance to nearest 0-cell in a grid, nearest city, etc.
# ──────────────────────────────────────────────

def multi_source_bfs(graph: defaultdict, sources: List[int], n: int) -> List[int]:
    """
    Returns dist[] where dist[v] = minimum distance from any source to v.
    dist[v] = -1 if v is unreachable from all sources.
    """
    INF = -1
    dist = [INF] * n
    queue: deque = deque()
    for s in sources:
        dist[s] = 0
        queue.append(s)
    while queue:
        node = queue.popleft()
        for nb in graph[node]:
            if dist[nb] == INF:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


# ──────────────────────────────────────────────
# 4. BFS Level-by-Level (useful when you need to process each level separately)
# ──────────────────────────────────────────────

def bfs_levels(graph: defaultdict, start: int) -> List[List[int]]:
    """Returns list of levels; levels[k] = all nodes at distance k from start."""
    visited = {start}
    queue = deque([start])
    levels = []
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node)
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        levels.append(level)
    return levels


# ──────────────────────────────────────────────
# 5. 0-1 BFS (deque-based BFS for graphs with edge weights 0 or 1)
#    Use instead of Dijkstra when all edges are 0 or 1.
# ──────────────────────────────────────────────

def bfs_01(graph: defaultdict, src: int, n: int) -> List[int]:
    """
    0-1 BFS: graph edges are (neighbor, weight) where weight ∈ {0, 1}.
    Returns dist[] array. Weight-0 edges go to front of deque, weight-1 to back.
    """
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0
    dq: deque = deque([src])
    while dq:
        node = dq.popleft()
        for nb, w in graph[node]:
            new_d = dist[node] + w
            if new_d < dist[nb]:
                dist[nb] = new_d
                if w == 0:
                    dq.appendleft(nb)
                else:
                    dq.append(nb)
    return dist


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Unweighted graph: 0-1-2-3, also 0-2
    edges = [(0, 1), (1, 2), (2, 3), (0, 2)]
    g = build_graph(4, edges)

    print("BFS order from 0:", bfs(g, 0))             # [0, 1, 2, 3]
    print("BFS levels from 0:", bfs_levels(g, 0))     # [[0], [1,2], [3]]

    dist, par = bfs_shortest_path(g, 0, 4)
    print("Distances from 0:", dist)                   # [0, 1, 1, 2]
    print("Path 0->3:", reconstruct_path(par, 0, 3))  # [0, 2, 3] or [0, 1, 2, 3]

    # Multi-source BFS from nodes 0 and 3
    print("Multi-source BFS [0,3]:", multi_source_bfs(g, [0, 3], 4))  # [0, 1, 1, 0]

    # 0-1 BFS example: edges with weight 0 or 1
    g01: defaultdict = defaultdict(list)
    g01[0].extend([(1, 0), (2, 1)])
    g01[1].extend([(0, 0), (3, 1)])
    g01[2].extend([(0, 1), (3, 0)])
    g01[3].extend([(1, 1), (2, 0)])
    print("0-1 BFS from 0:", bfs_01(g01, 0, 4))      # [0, 0, 1, 1]
