"""
NAME: Depth-First Search (DFS)
TAGS: graph, dfs, traversal, connected-components, cycle-detection
DESCRIPTION: Explores a graph by going as deep as possible before backtracking. Use for
    connected component counting, cycle detection, topological ordering, and reachability.
    Works on both directed and undirected graphs.
COMPLEXITY: Time O(V + E), Space O(V)
"""

from collections import defaultdict
from typing import List, Optional


# ──────────────────────────────────────────────
# Graph representation helpers
# ──────────────────────────────────────────────

def build_graph(n: int, edges: List[tuple], directed: bool = False) -> defaultdict:
    g = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
        if not directed:
            g[v].append(u)
    return g


# ──────────────────────────────────────────────
# 1. Recursive DFS
# ──────────────────────────────────────────────

def dfs_recursive(graph: defaultdict, start: int, visited: Optional[set] = None) -> List[int]:
    """Returns list of nodes visited from `start` in DFS order."""
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for nb in graph[start]:
        if nb not in visited:
            order.extend(dfs_recursive(graph, nb, visited))
    return order


# ──────────────────────────────────────────────
# 2. Iterative DFS (avoids recursion limit)
# ──────────────────────────────────────────────

def dfs_iterative(graph: defaultdict, start: int) -> List[int]:
    """Returns list of nodes visited from `start` in DFS order (iterative)."""
    visited = set()
    stack = [start]
    order = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # Push in reverse order to maintain left-to-right traversal
        for nb in reversed(graph[node]):
            if nb not in visited:
                stack.append(nb)
    return order


# ──────────────────────────────────────────────
# 3. Connected Components (undirected graph)
# ──────────────────────────────────────────────

def connected_components(n: int, graph: defaultdict) -> List[List[int]]:
    """
    Returns all connected components as lists of node indices.
    Nodes are assumed to be labeled 0..n-1.
    """
    visited = set()
    components = []

    def _dfs(node: int, comp: list):
        visited.add(node)
        comp.append(node)
        for nb in graph[node]:
            if nb not in visited:
                _dfs(nb, comp)

    for i in range(n):
        if i not in visited:
            comp: List[int] = []
            _dfs(i, comp)
            components.append(comp)
    return components


def count_components(n: int, graph: defaultdict) -> int:
    """Returns the number of connected components."""
    return len(connected_components(n, graph))


# ──────────────────────────────────────────────
# 4. Cycle Detection — Undirected Graph
#    A back-edge (edge to a visited node that isn't the parent) means cycle.
# ──────────────────────────────────────────────

def has_cycle_undirected(n: int, graph: defaultdict) -> bool:
    """Returns True if the undirected graph (0-indexed, n nodes) contains a cycle."""
    visited = set()

    def _dfs(node: int, parent: int) -> bool:
        visited.add(node)
        for nb in graph[node]:
            if nb not in visited:
                if _dfs(nb, node):
                    return True
            elif nb != parent:          # back-edge found
                return True
        return False

    for i in range(n):
        if i not in visited:
            if _dfs(i, -1):
                return True
    return False


# ──────────────────────────────────────────────
# 5. DFS Helper: find a path between two nodes
# ──────────────────────────────────────────────

def find_path(graph: defaultdict, src: int, dst: int) -> Optional[List[int]]:
    """Returns a path from src to dst, or None if unreachable."""
    visited = set()

    def _dfs(node: int, path: list) -> Optional[List[int]]:
        if node == dst:
            return path[:]
        visited.add(node)
        for nb in graph[node]:
            if nb not in visited:
                path.append(nb)
                result = _dfs(nb, path)
                if result is not None:
                    return result
                path.pop()
        return None

    return _dfs(src, [src])


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Graph:  0-1-2   3-4   (two components)
    #             |
    #             2
    edges = [(0, 1), (1, 2), (3, 4)]
    g = build_graph(5, edges)

    print("Recursive DFS from 0:", dfs_recursive(g, 0))        # [0, 1, 2]
    print("Iterative DFS from 0:", dfs_iterative(g, 0))         # [0, 1, 2]
    print("Components:", connected_components(5, g))             # [[0,1,2], [3,4]]
    print("Cycle in undirected?", has_cycle_undirected(5, g))   # False

    # Add edge to create a cycle: 0-1-2-0
    g2 = build_graph(3, [(0, 1), (1, 2), (2, 0)])
    print("Cycle after adding 2-0?", has_cycle_undirected(3, g2))  # True

    print("Path 0->2:", find_path(g, 0, 2))   # [0, 1, 2]
    print("Path 0->4:", find_path(g, 0, 4))   # None
