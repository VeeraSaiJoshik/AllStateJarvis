"""
NAME: Topological Sort
TAGS: graph, dag, topological-sort, kahns, dfs, ordering, directed
DESCRIPTION: Produces a linear ordering of vertices in a DAG such that for every
    directed edge u->v, u appears before v. Use for dependency resolution, scheduling,
    build systems, and course prerequisites. Two implementations: Kahn's (BFS) and DFS.
COMPLEXITY: Time O(V + E), Space O(V)
"""

from collections import defaultdict, deque
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────
# Graph helper
# ──────────────────────────────────────────────

def build_graph(n: int, edges: List[Tuple[int, int]]) -> defaultdict:
    """Builds directed adjacency list. Nodes 0-indexed."""
    g: defaultdict = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
    return g


# ──────────────────────────────────────────────
# 1. Kahn's Algorithm (BFS-based)
#    Repeatedly remove nodes with in-degree 0.
#    If result doesn't include all nodes → cycle exists (not a DAG).
# ──────────────────────────────────────────────

def topological_sort_kahn(n: int, graph: defaultdict) -> Optional[List[int]]:
    """
    Topological sort using Kahn's algorithm.
    Returns the topological order, or None if the graph has a cycle.
    Nodes 0-indexed, n total.
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1

    # Start with all nodes that have in-degree 0
    queue: deque = deque(i for i in range(n) if in_degree[i] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return order if len(order) == n else None   # None → cycle detected


# ──────────────────────────────────────────────
# 2. DFS-based Topological Sort
#    Post-order DFS; reverse the result.
# ──────────────────────────────────────────────

def topological_sort_dfs(n: int, graph: defaultdict) -> Optional[List[int]]:
    """
    Topological sort using DFS post-order.
    Returns the topological order, or None if the graph has a cycle.
    Uses three-color marking: 0=unvisited, 1=in-stack, 2=done.
    """
    color = [0] * n   # 0: white, 1: gray (in stack), 2: black (done)
    order = []
    has_cycle = False

    def _dfs(u: int):
        nonlocal has_cycle
        if has_cycle:
            return
        color[u] = 1          # mark as in-stack
        for v in graph[u]:
            if color[v] == 1:
                has_cycle = True   # back edge → cycle
                return
            if color[v] == 0:
                _dfs(v)
        color[u] = 2          # mark as fully processed
        order.append(u)       # post-order

    for i in range(n):
        if color[i] == 0:
            _dfs(i)
            if has_cycle:
                return None

    order.reverse()
    return order


# ──────────────────────────────────────────────
# 3. Lexicographically Smallest Topological Order
#    Use a min-heap instead of a regular queue in Kahn's.
# ──────────────────────────────────────────────

import heapq

def topological_sort_lex(n: int, graph: defaultdict) -> Optional[List[int]]:
    """
    Returns lexicographically smallest topological order using a min-heap.
    Returns None if graph has a cycle.
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1

    heap = [i for i in range(n) if in_degree[i] == 0]
    heapq.heapify(heap)
    order = []

    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(heap, v)

    return order if len(order) == n else None


# ──────────────────────────────────────────────
# 4. All Topological Orderings (backtracking)
#    WARNING: exponential time — only use for small graphs (n ≤ 8).
# ──────────────────────────────────────────────

def all_topological_sorts(n: int, graph: defaultdict) -> List[List[int]]:
    """Returns all valid topological orderings of the DAG."""
    in_degree = [0] * n
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1

    result = []
    current: List[int] = []
    visited = [False] * n

    def _backtrack():
        if len(current) == n:
            result.append(current[:])
            return
        for u in range(n):
            if not visited[u] and in_degree[u] == 0:
                visited[u] = True
                current.append(u)
                for v in graph[u]:
                    in_degree[v] -= 1
                _backtrack()
                # Undo
                for v in graph[u]:
                    in_degree[v] += 1
                current.pop()
                visited[u] = False

    _backtrack()
    return result


# ──────────────────────────────────────────────
# 5. Longest Path in DAG (DP on topological order)
# ──────────────────────────────────────────────

def dag_longest_path(n: int, graph: defaultdict) -> int:
    """
    Returns the length (in edges) of the longest path in a DAG.
    Uses topological sort + DP.
    """
    order = topological_sort_kahn(n, graph)
    if order is None:
        raise ValueError("Graph has a cycle — not a DAG")

    dp = [0] * n
    for u in order:
        for v in graph[u]:
            dp[v] = max(dp[v], dp[u] + 1)

    return max(dp)


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # DAG: 5->2, 5->0, 4->0, 4->1, 2->3, 3->1
    edges = [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)]
    g = build_graph(6, edges)

    print("Kahn's sort:", topological_sort_kahn(6, g))
    # One valid order: [4, 5, 2, 0, 3, 1]

    print("DFS sort:", topological_sort_dfs(6, g))
    # One valid order: [5, 4, 2, 3, 1, 0] or similar

    print("Lex sort:", topological_sort_lex(6, g))
    # [4, 5, 0, 2, 3, 1]

    # Cycle detection
    cyclic_edges = [(0,1),(1,2),(2,0)]
    cg = build_graph(3, cyclic_edges)
    print("Cycle detected (Kahn):", topological_sort_kahn(3, cg) is None)   # True
    print("Cycle detected (DFS):", topological_sort_dfs(3, cg) is None)     # True

    # Longest path
    g2 = build_graph(6, edges)
    print("Longest path in DAG:", dag_longest_path(6, g2))  # 3 (5->2->3->1)
