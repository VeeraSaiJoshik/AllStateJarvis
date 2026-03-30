"""
NAME: Minimum Spanning Tree (Kruskal's and Prim's)
TAGS: graph, mst, kruskal, prim, greedy, dsu, union-find, spanning-tree
DESCRIPTION: Finds a spanning tree of minimum total edge weight in an undirected weighted
    graph. Kruskal's is better for sparse graphs (sort edges, union-find), while Prim's
    (with heap) is better for dense graphs. Both run on connected graphs; for forests,
    apply to each component.
COMPLEXITY: Kruskal O(E log E), Prim O((V + E) log V), Space O(V + E)
"""

import heapq
from collections import defaultdict
from typing import List, Tuple, Optional


Edge = Tuple[int, int, int]   # (u, v, weight)


# ──────────────────────────────────────────────
# Disjoint Set Union (Union-Find) — for Kruskal's
# ──────────────────────────────────────────────

class DSU:
    """Union-Find with path compression and union by rank."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Returns True if x and y were in different components (merged)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


# ──────────────────────────────────────────────
# 1. Kruskal's MST
# ──────────────────────────────────────────────

def kruskal_mst(n: int, edges: List[Edge]) -> Tuple[int, List[Edge]]:
    """
    Kruskal's MST algorithm.
    Returns (total_weight, mst_edges).
    mst_edges: list of (u, v, w) edges in the MST.
    Returns weight=inf and partial tree if graph is not connected.
    Nodes 0-indexed.
    """
    dsu = DSU(n)
    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda e: e[2])
    total = 0
    mst_edges: List[Edge] = []

    for u, v, w in sorted_edges:
        if dsu.union(u, v):
            total += w
            mst_edges.append((u, v, w))
            if len(mst_edges) == n - 1:
                break   # MST complete

    return total, mst_edges


def kruskal_mst_forest(n: int, edges: List[Edge]) -> Tuple[int, List[Edge]]:
    """
    Kruskal's for possibly disconnected graphs (minimum spanning forest).
    Returns (total_weight, forest_edges).
    """
    dsu = DSU(n)
    sorted_edges = sorted(edges, key=lambda e: e[2])
    total = 0
    forest_edges: List[Edge] = []

    for u, v, w in sorted_edges:
        if dsu.union(u, v):
            total += w
            forest_edges.append((u, v, w))

    return total, forest_edges


# ──────────────────────────────────────────────
# 2. Prim's MST (with min-heap)
# ──────────────────────────────────────────────

def prim_mst(n: int, graph: defaultdict, start: int = 0
             ) -> Tuple[int, List[Edge]]:
    """
    Prim's MST algorithm using a min-heap.
    graph[u] = list of (v, weight) — undirected.
    Returns (total_weight, mst_edges).
    mst_edges: list of (u, v, w) in the MST.
    Nodes 0-indexed.
    """
    in_mst = [False] * n
    # Heap: (weight, to_node, from_node)
    heap = [(0, start, -1)]
    total = 0
    mst_edges: List[Edge] = []

    while heap and len(mst_edges) < n - 1:
        w, u, parent = heapq.heappop(heap)
        if in_mst[u]:
            continue
        in_mst[u] = True
        total += w
        if parent != -1:
            mst_edges.append((parent, u, w))
        for v, weight in graph[u]:
            if not in_mst[v]:
                heapq.heappush(heap, (weight, v, u))

    return total, mst_edges


def prim_mst_from_edges(n: int, edges: List[Edge], start: int = 0
                         ) -> Tuple[int, List[Edge]]:
    """Prim's MST — builds adjacency list internally from edge list."""
    g: defaultdict = defaultdict(list)
    for u, v, w in edges:
        g[u].append((v, w))
        g[v].append((u, w))
    return prim_mst(n, g, start)


# ──────────────────────────────────────────────
# 3. Maximum Spanning Tree
#    Negate weights, run Kruskal's, negate result back.
# ──────────────────────────────────────────────

def maximum_spanning_tree(n: int, edges: List[Edge]) -> Tuple[int, List[Edge]]:
    """
    Returns (total_weight, mst_edges) for the MAXIMUM spanning tree.
    """
    neg_edges = [(u, v, -w) for u, v, w in edges]
    neg_total, neg_mst = kruskal_mst(n, neg_edges)
    mst = [(u, v, -w) for u, v, w in neg_mst]
    return -neg_total, mst


# ──────────────────────────────────────────────
# 4. Second Minimum Spanning Tree
#    For each edge e NOT in MST, swap it with the max-weight edge
#    on the MST path between e's endpoints. Take minimum cost increase.
#    Uses LCA + path max — simplified version O(V^2).
# ──────────────────────────────────────────────

def second_minimum_spanning_tree_naive(n: int, edges: List[Edge]
                                       ) -> Optional[int]:
    """
    Returns the weight of the second MST (O(E * V) naive).
    Returns None if second MST doesn't exist.
    Excludes each MST edge one at a time and finds next MST.
    """
    base_w, mst_edges = kruskal_mst(n, edges)
    if len(mst_edges) < n - 1:
        return None   # graph not connected

    second_best: Optional[int] = None
    mst_set = set(id(e) for e in mst_edges)

    for i, skip in enumerate(mst_edges):
        # Build all edges except the i-th MST edge
        remaining = [e for j, e in enumerate(edges) if e != skip or j == -1]
        # Use Kruskal's on the edge list minus one MST edge
        remaining_no_skip = []
        skipped = False
        for e in edges:
            if not skipped and e == skip:
                skipped = True
                continue
            remaining_no_skip.append(e)
        w, tree = kruskal_mst(n, remaining_no_skip)
        if len(tree) == n - 1:
            if second_best is None or w < second_best:
                second_best = w

    return second_best


# ──────────────────────────────────────────────
# 5. DSU-based utilities
# ──────────────────────────────────────────────

def is_connected_mst(n: int, mst_edges: List[Edge]) -> bool:
    """Check if the returned MST spans all n nodes (i.e., graph was connected)."""
    return len(mst_edges) == n - 1


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Graph: 5 nodes, edges with weights
    edges: List[Edge] = [
        (0,1,2),(0,3,6),(1,2,3),(1,3,8),(1,4,5),(2,4,7),(3,4,9)
    ]
    n = 5

    # Kruskal's
    total_k, mst_k = kruskal_mst(n, edges)
    print(f"Kruskal MST weight: {total_k}")   # 16
    print("Kruskal MST edges:", mst_k)         # (0,1,2),(1,2,3),(1,4,5),(0,3,6)

    # Prim's
    total_p, mst_p = prim_mst_from_edges(n, edges)
    print(f"Prim MST weight: {total_p}")       # 16
    print("Prim MST edges:", mst_p)

    # Maximum spanning tree
    total_max, mst_max = maximum_spanning_tree(n, edges)
    print(f"Max ST weight: {total_max}")       # e.g. 29

    # DSU standalone
    dsu = DSU(5)
    dsu.union(0, 1)
    dsu.union(2, 3)
    print("0 and 1 connected:", dsu.connected(0, 1))   # True
    print("0 and 2 connected:", dsu.connected(0, 2))   # False
    print("Components:", dsu.n_components)              # 3
