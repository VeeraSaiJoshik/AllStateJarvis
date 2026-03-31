"""
NAME: Tree DP Patterns (Subtree Sizes, Rerooting, Diameter, Centroid Decomposition)
TAGS: tree DP, rerooting, diameter, centroid decomposition, subtree, dynamic programming
DESCRIPTION: Essential tree DP patterns for competitive programming: subtree size/sum DP,
    rerooting technique for computing values rooted at every node in O(n), tree diameter
    via two-pass DFS, and centroid decomposition for path problems across the tree.
COMPLEXITY: All O(n) or O(n log n) for centroid decomposition
"""

import sys
from typing import List, Tuple, Callable, Optional

sys.setrecursionlimit(300000)


# ─────────────────────────────────────────────
# 1. Subtree sizes and basic tree DP
# ─────────────────────────────────────────────
def subtree_sizes(n: int, adj: List[List[int]], root: int = 0) -> List[int]:
    """Return subtree size for each node (iterative DFS)."""
    size = [1] * n
    parent = [-1] * n
    order: List[int] = []

    # Iterative BFS to get topological order
    from collections import deque
    q = deque([root])
    visited = [False] * n
    visited[root] = True
    while q:
        v = q.popleft()
        order.append(v)
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                parent[u] = v
                q.append(u)

    # Process in reverse BFS order (leaves first)
    for v in reversed(order):
        if parent[v] != -1:
            size[parent[v]] += size[v]

    return size


# ─────────────────────────────────────────────
# 2. Rerooting Technique
# Compute ans[v] = f(tree rooted at v) for all v in O(n)
# Generic pattern: first pass (root=0), then reroot pass
# ─────────────────────────────────────────────
def rerooting_sum_depths(n: int, adj: List[List[int]]) -> List[int]:
    """
    Classic rerooting: ans[v] = sum of distances from v to all other nodes.
    ans[0] via DFS, then ans[child] = ans[parent] - size[child] + (n - size[child])
    """
    from collections import deque

    size = [1] * n
    depth_sum = [0] * n
    parent = [-1] * n
    order: List[int] = []

    q = deque([0])
    visited = [False] * n
    visited[0] = True
    while q:
        v = q.popleft()
        order.append(v)
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                parent[u] = v
                q.append(u)

    # Up-pass: compute depth_sum[0] and subtree sizes
    depth = [0] * n
    for v in reversed(order):
        if parent[v] != -1:
            size[parent[v]] += size[v]
            depth_sum[parent[v]] += depth_sum[v] + size[v]

    # depth_sum[0] now = sum of depths from root 0
    ans = depth_sum[:]

    # Down-pass: reroot
    for v in order:
        for u in adj[v]:
            if u != parent[v]:
                # ans[u] = ans[v] - size[u]  (u's subtree contributes 1 less depth)
                #                 + (n - size[u])  (rest contributes 1 more depth)
                ans[u] = ans[v] - size[u] + (n - size[u])

    return ans


# ─────────────────────────────────────────────
# 3. Tree Diameter
# ─────────────────────────────────────────────
def tree_diameter(n: int, adj: List[List[Tuple[int, int]]]) -> Tuple[int, int, int]:
    """
    Find tree diameter (longest path) using two BFS passes.
    adj[v] = list of (neighbor, weight).
    Returns (diameter_length, endpoint_u, endpoint_v).
    """
    from collections import deque

    def bfs_farthest(src: int) -> Tuple[int, List[int]]:
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        farthest = src
        while q:
            v = q.popleft()
            for u, w in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + w
                    if dist[u] > dist[farthest]:
                        farthest = u
                    q.append(u)
        return farthest, dist

    # First BFS from node 0
    u, _ = bfs_farthest(0)
    # Second BFS from farthest node u
    v, dist = bfs_farthest(u)
    return dist[v], u, v


def tree_diameter_dp(n: int, adj: List[List[int]]) -> int:
    """
    Diameter via DP on subtrees (unweighted). O(n).
    For each node, diameter through it = two longest arms.
    """
    from collections import deque

    depth = [0] * n
    parent = [-1] * n
    order: List[int] = []
    visited = [False] * n
    q = deque([0])
    visited[0] = True
    while q:
        v = q.popleft()
        order.append(v)
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                parent[u] = v
                q.append(u)

    diameter = 0
    # max depth in subtree
    max_d = [0] * n
    for v in reversed(order):
        children_depths = [max_d[u] + 1 for u in adj[v] if u != parent[v]]
        if not children_depths:
            continue
        children_depths.sort(reverse=True)
        # Diameter through v = top two child arms
        if len(children_depths) >= 2:
            diameter = max(diameter, children_depths[0] + children_depths[1])
        else:
            diameter = max(diameter, children_depths[0])
        max_d[v] = children_depths[0]

    return diameter


# ─────────────────────────────────────────────
# 4. Centroid Decomposition
# ─────────────────────────────────────────────
class CentroidDecomposition:
    """
    Decomposes tree into O(log n) centroid layers.
    Use for counting/finding paths with certain properties.
    Template: override solve(centroid, component_nodes).
    """

    def __init__(self, n: int, adj: List[List[int]]):
        self.n = n
        self.adj = adj
        self.size = [0] * n
        self.removed = [False] * n
        self.centroid_parent = [-1] * n

        # Build centroid decomposition tree
        self._decompose(0, -1)

    def _get_subtree_size(self, v: int, parent: int) -> int:
        self.size[v] = 1
        for u in self.adj[v]:
            if u != parent and not self.removed[u]:
                self.size[v] += self._get_subtree_size(u, v)
        return self.size[v]

    def _get_centroid(self, v: int, parent: int, tree_size: int) -> int:
        for u in self.adj[v]:
            if u != parent and not self.removed[u]:
                if self.size[u] > tree_size // 2:
                    return self._get_centroid(u, v, tree_size)
        return v

    def _decompose(self, v: int, par_centroid: int) -> None:
        sz = self._get_subtree_size(v, -1)
        c = self._get_centroid(v, -1, sz)
        self.centroid_parent[c] = par_centroid
        self.removed[c] = True
        for u in self.adj[c]:
            if not self.removed[u]:
                self._decompose(u, c)

    def get_path_through_ancestor(self, v: int):
        """
        Yield v, then each centroid ancestor of v in order.
        Use to iterate over all centroids that contain a path through v.
        """
        node = v
        while node != -1:
            yield node
            node = self.centroid_parent[node]


# ─────────────────────────────────────────────
# 5. Heavy-Light Decomposition (HLD)
# ─────────────────────────────────────────────
class HLD:
    """
    Heavy-Light Decomposition: decomposes tree paths into O(log n) chains.
    After decomposition, path queries reduce to O(log^2 n) range queries on an array.
    """

    def __init__(self, n: int, adj: List[List[int]], root: int = 0):
        self.n = n
        self.adj = adj
        self.root = root
        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.heavy = [-1] * n       # heavy child
        self.head = [0] * n         # chain head
        self.pos = [0] * n          # position in flattened array
        self._build()

    def _build(self) -> None:
        from collections import deque

        # BFS to set parent/depth
        order: List[int] = []
        q = deque([self.root])
        visited = [False] * self.n
        visited[self.root] = True
        while q:
            v = q.popleft()
            order.append(v)
            for u in self.adj[v]:
                if not visited[u]:
                    visited[u] = True
                    self.parent[u] = v
                    self.depth[u] = self.depth[v] + 1
                    q.append(u)

        # Bottom-up: compute sizes and heavy children
        for v in reversed(order):
            max_sz, heavy_child = 0, -1
            for u in self.adj[v]:
                if u != self.parent[v]:
                    self.size[v] += self.size[u]
                    if self.size[u] > max_sz:
                        max_sz = self.size[u]
                        heavy_child = u
            self.heavy[v] = heavy_child

        # Top-down: assign chain heads and positions
        timer = [0]
        def assign(v: int, h: int) -> None:
            self.head[v] = h
            self.pos[v] = timer[0]
            timer[0] += 1
            if self.heavy[v] != -1:
                assign(self.heavy[v], h)
            for u in self.adj[v]:
                if u != self.parent[v] and u != self.heavy[v]:
                    assign(u, u)

        assign(self.root, self.root)

    def path_query(self, u: int, v: int, query_fn: Callable[[int, int], int]) -> int:
        """
        Query over path u→v using query_fn(l, r) on the flat array.
        Returns combined result (assumes query_fn is sum or min/max).
        Pass your segment tree's range query as query_fn.
        """
        result = 0
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            result += query_fn(self.pos[self.head[u]], self.pos[u])
            u = self.parent[self.head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result += query_fn(self.pos[u], self.pos[v])
        return result


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    #   0 - 1 - 2
    #       |
    #       3 - 4
    n = 5
    adj = [[] for _ in range(n)]
    for u, v in [(0,1),(1,2),(1,3),(3,4)]:
        adj[u].append(v)
        adj[v].append(u)

    # Subtree sizes
    sz = subtree_sizes(n, adj, root=0)
    assert sz[0] == 5
    assert sz[1] == 4
    assert sz[2] == 1

    # Rerooting: sum of distances
    ans = rerooting_sum_depths(n, adj)
    # Node 1: distances = 1+1+2+2=6? Let's verify node 0: 0-1=1, 0-2=2, 0-3=2, 0-4=3 => 8
    assert ans[0] == 8

    # Diameter (unweighted)
    d = tree_diameter_dp(n, adj)
    assert d == 3   # path 0-1-3-4 or 2-1-3-4

    # Weighted diameter
    wadj = [[] for _ in range(n)]
    for u, v in [(0,1),(1,2),(1,3),(3,4)]:
        wadj[u].append((v, 1))
        wadj[v].append((u, 1))
    diam, eu, ev = tree_diameter(n, wadj)
    assert diam == 3

    # Centroid decomposition
    cd = CentroidDecomposition(n, adj)
    # centroid of whole tree should be node 1 or 3
    # Just verify it builds without error
    path = list(cd.get_path_through_ancestor(4))
    assert 4 in path

    print("All tree DP tests passed.")
