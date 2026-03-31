"""
NAME: Floyd-Warshall All-Pairs Shortest Path
TAGS: graph, all-pairs, shortest-path, dynamic-programming, negative-weights, dense
DESCRIPTION: Computes shortest paths between every pair of vertices in a weighted graph,
    handling negative edge weights. Use for dense graphs or when you need all-pairs
    distances; also detects negative cycles via negative diagonal values.
COMPLEXITY: Time O(V^3), Space O(V^2)
"""

from typing import List, Tuple, Optional


INF = float('inf')


# ──────────────────────────────────────────────
# 1. Standard Floyd-Warshall
#    Input: n nodes (0-indexed), weighted edge list (u, v, w).
#    Returns: dist[n][n] matrix.
#    dist[i][j] = shortest path from i to j; inf if unreachable.
#    Negative cycles: if dist[i][i] < 0 for any i.
# ──────────────────────────────────────────────

def floyd_warshall(n: int, edges: List[Tuple[int, int, float]]
                   ) -> List[List[float]]:
    """
    All-pairs shortest paths.
    edges: list of (u, v, weight) — directed edges.
    Self-loops and multiple edges are handled correctly.
    """
    # Initialize distance matrix
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        # Keep minimum weight if multiple edges exist
        if w < dist[u][v]:
            dist[u][v] = w

    # Relax through each intermediate vertex k
    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue                 # prune: no path i->k
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


# ──────────────────────────────────────────────
# 2. Negative cycle detection
# ──────────────────────────────────────────────

def has_negative_cycle(dist: List[List[float]]) -> bool:
    """Returns True if the dist matrix (from floyd_warshall) contains a negative cycle."""
    n = len(dist)
    return any(dist[i][i] < 0 for i in range(n))


# ──────────────────────────────────────────────
# 3. Floyd-Warshall with path reconstruction
#    next_node[i][j] = first step from i toward j on the shortest path.
# ──────────────────────────────────────────────

def floyd_warshall_with_path(n: int, edges: List[Tuple[int, int, float]]
                              ) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Returns (dist, nxt) where nxt[i][j] = next node on shortest path from i to j.
    nxt[i][j] = -1 means no path exists.
    """
    dist = [[INF] * n for _ in range(n)]
    nxt = [[-1] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
        nxt[i][i] = i

    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
            nxt[u][v] = v

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    nxt[i][j] = nxt[i][k]

    return dist, nxt


def reconstruct_path(nxt: List[List[int]], src: int, dst: int) -> Optional[List[int]]:
    """
    Reconstruct shortest path from src to dst using the next-hop matrix.
    Returns None if dst is unreachable from src.
    """
    if nxt[src][dst] == -1:
        return None
    path = [src]
    node = src
    while node != dst:
        node = nxt[node][dst]
        path.append(node)
        if len(path) > len(nxt):          # cycle guard
            return None
    return path


# ──────────────────────────────────────────────
# 4. Floyd-Warshall on adjacency matrix input
#    (alternative interface — convenient for dense graph input)
# ──────────────────────────────────────────────

def floyd_warshall_matrix(adj: List[List[float]]) -> List[List[float]]:
    """
    All-pairs shortest paths given an adjacency matrix adj[i][j].
    adj[i][j] should be INF (float('inf')) if no direct edge.
    adj[i][i] should be 0.
    Modifies and returns adj in-place.
    """
    n = len(adj)
    for k in range(n):
        for i in range(n):
            if adj[i][k] == INF:
                continue
            for j in range(n):
                if adj[i][k] + adj[k][j] < adj[i][j]:
                    adj[i][j] = adj[i][k] + adj[k][j]
    return adj


# ──────────────────────────────────────────────
# 5. Transitive Closure (reachability matrix)
#    Use bitwise OR instead of min — much faster for large n.
# ──────────────────────────────────────────────

def transitive_closure(n: int, edges: List[Tuple[int, int]]) -> List[List[bool]]:
    """
    Returns reach[i][j] = True if there is a path from i to j.
    edges: unweighted directed edge list (u, v).
    """
    reach = [[False] * n for _ in range(n)]
    for i in range(n):
        reach[i][i] = True
    for u, v in edges:
        reach[u][v] = True

    for k in range(n):
        for i in range(n):
            if not reach[i][k]:
                continue
            for j in range(n):
                if reach[k][j]:
                    reach[i][j] = True
    return reach


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Directed graph: 0->1(3), 0->3(7), 1->0(8), 1->2(2),
    #                 2->0(5), 2->3(1), 3->0(2)
    edges = [(0,1,3),(0,3,7),(1,0,8),(1,2,2),(2,0,5),(2,3,1),(3,0,2)]
    n = 4

    dist = floyd_warshall(n, edges)
    print("All-pairs distances:")
    for row in dist:
        print(row)
    # Expected: [0,3,5,6], [8,0,2,3], [5,8,0,1], [2,5,7,0]

    print("Has negative cycle:", has_negative_cycle(dist))   # False

    dist2, nxt = floyd_warshall_with_path(n, edges)
    print("Path 0->2:", reconstruct_path(nxt, 0, 2))         # [0, 1, 2]
    print("Path 2->0:", reconstruct_path(nxt, 2, 0))         # [2, 3, 0]

    # Transitive closure
    reach = transitive_closure(4, [(u, v) for u, v, _ in edges])
    print("Reachable from 0:", [j for j in range(4) if reach[0][j]])  # [0,1,2,3]
