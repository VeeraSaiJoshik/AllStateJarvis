"""
NAME: Bellman-Ford Shortest Path Algorithm
TAGS: graph, shortest-path, negative-weights, negative-cycle, sssp, dynamic-programming
DESCRIPTION: Finds single-source shortest paths in graphs that may have negative edge
    weights, and detects negative-weight cycles. Use when edge weights can be negative
    and Dijkstra is not applicable; also used to detect arbitrage in currency exchange.
COMPLEXITY: Time O(V * E), Space O(V)
"""

from collections import defaultdict
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────
# Edge representation: list of (u, v, weight)
# ──────────────────────────────────────────────

Edge = Tuple[int, int, float]


# ──────────────────────────────────────────────
# 1. Standard Bellman-Ford — SSSP
#    Runs V-1 relaxation passes, then one more to detect negative cycles.
# ──────────────────────────────────────────────

def bellman_ford(n: int, edges: List[Edge], src: int
                 ) -> Tuple[List[float], bool]:
    """
    Single-source shortest paths from src.
    Returns (dist, has_negative_cycle).
    dist[v] = shortest distance from src to v; float('inf') if unreachable.
    has_negative_cycle = True if any negative cycle is reachable from src.
    Nodes 0-indexed, n total nodes.
    """
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0

    # Relax all edges V-1 times
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:           # early termination
            break

    # V-th pass: if any relaxation still possible → negative cycle
    has_neg_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_neg_cycle = True
            break

    return dist, has_neg_cycle


# ──────────────────────────────────────────────
# 2. Bellman-Ford with parent tracking
# ──────────────────────────────────────────────

def bellman_ford_with_path(n: int, edges: List[Edge], src: int
                           ) -> Tuple[List[float], List[int], bool]:
    """
    Returns (dist, parent, has_negative_cycle).
    parent[v] = predecessor of v on shortest path (-1 if src or unreachable).
    """
    INF = float('inf')
    dist = [INF] * n
    parent = [-1] * n
    dist[src] = 0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            break

    has_neg_cycle = any(
        dist[u] != INF and dist[u] + w < dist[v]
        for u, v, w in edges
    )
    return dist, parent, has_neg_cycle


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
# 3. Detect ALL nodes affected by a negative cycle
#    (those whose shortest path is −∞)
# ──────────────────────────────────────────────

def bellman_ford_neg_inf(n: int, edges: List[Edge], src: int) -> List[float]:
    """
    Returns dist[] where dist[v] = -inf if v is reachable via a negative cycle,
    inf if unreachable, or the actual shortest distance otherwise.
    Useful when you need to output -infinity for nodes on/after negative cycles.
    """
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Mark nodes reachable via negative cycles
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = -INF           # propagate -inf

    return dist


# ──────────────────────────────────────────────
# 4. Negative cycle detection only (global — any cycle in graph)
#    Start with dist[all] = 0 to detect even unreachable cycles.
# ──────────────────────────────────────────────

def has_any_negative_cycle(n: int, edges: List[Edge]) -> bool:
    """
    Returns True if the graph contains ANY negative cycle (not just from src).
    Trick: initialize all distances to 0 (as if a virtual source connects everywhere).
    """
    dist = [0.0] * n
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return any(dist[u] + w < dist[v] for u, v, w in edges)


# ──────────────────────────────────────────────
# 5. SPFA (Shortest Path Faster Algorithm)
#    Queue-based Bellman-Ford; faster in practice, same worst case.
#    Negative cycle detection via visit count per node.
# ──────────────────────────────────────────────

from collections import deque

def spfa(n: int, graph: defaultdict, src: int) -> Tuple[List[float], bool]:
    """
    SPFA: queue-optimized Bellman-Ford.
    graph[u] = list of (v, w).
    Returns (dist, has_negative_cycle).
    """
    INF = float('inf')
    dist = [INF] * n
    in_queue = [False] * n
    relax_count = [0] * n     # times node was relaxed; >= n means neg cycle

    dist[src] = 0
    queue: deque = deque([src])
    in_queue[src] = True

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                relax_count[v] += 1
                if relax_count[v] >= n:
                    return dist, True    # negative cycle detected
                if not in_queue[v]:
                    in_queue[v] = True
                    queue.append(v)

    return dist, False


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Graph: 0->1 (1), 1->2 (-3), 2->3 (2), 0->3 (10)
    edges: List[Edge] = [(0, 1, 1), (1, 2, -3), (2, 3, 2), (0, 3, 10)]
    dist, neg = bellman_ford(4, edges, 0)
    print("Distances:", dist)          # [0, 1, -2, 0]
    print("Negative cycle:", neg)      # False

    # Negative cycle: 0->1 (1), 1->2 (-2), 2->0 (-1)
    neg_edges: List[Edge] = [(0, 1, 1), (1, 2, -2), (2, 0, -1)]
    dist2, neg2 = bellman_ford(3, neg_edges, 0)
    print("Negative cycle detected:", neg2)   # True

    # SPFA
    g = defaultdict(list)
    for u, v, w in edges:
        g[u].append((v, w))
    dist3, neg3 = spfa(4, g, 0)
    print("SPFA distances:", dist3)    # [0, 1, -2, 0]

    print("Any negative cycle:", has_any_negative_cycle(3, neg_edges))  # True
