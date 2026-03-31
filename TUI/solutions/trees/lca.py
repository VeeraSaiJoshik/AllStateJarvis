"""
NAME: Lowest Common Ancestor (LCA) — Binary Lifting and Euler Tour + Sparse Table
TAGS: LCA, binary lifting, sparse table, euler tour, tree, ancestor, graph
DESCRIPTION: Binary lifting preprocesses O(n log n) and answers LCA queries in O(log n)
    — also supports k-th ancestor queries. Euler tour + sparse table achieves O(1) query
    after O(n log n) preprocessing by reducing LCA to range minimum query on depths.
COMPLEXITY: Binary lifting: O(n log n) build, O(log n) query; Euler+Sparse: O(n log n) build, O(1) query
"""

import sys
from typing import List, Optional, Tuple

sys.setrecursionlimit(300000)


# ─────────────────────────────────────────────
# Method 1: Binary Lifting LCA
# ─────────────────────────────────────────────
class LCABinaryLifting:
    """
    Fast LCA using binary lifting (sparse table on ancestors).
    Also supports:
      - kth_ancestor(v, k): k-th ancestor of v
      - distance(u, v): edge distance between u and v
    """

    LOG = 18   # 2^18 > 2.6e5; set LOG=20 for n up to 10^6

    def __init__(self, n: int, root: int = 0):
        self.n = n
        self.root = root
        self.adj: List[List[int]] = [[] for _ in range(n)]
        self.depth = [0] * n
        self.up = [[-1] * n for _ in range(self.LOG)]

    def add_edge(self, u: int, v: int) -> None:
        self.adj[u].append(v)
        self.adj[v].append(u)

    def build(self) -> None:
        """Run BFS from root to fill depth and up tables."""
        from collections import deque
        q = deque([self.root])
        visited = [False] * self.n
        visited[self.root] = True
        self.up[0][self.root] = self.root   # root's parent is itself

        while q:
            v = q.popleft()
            for u in self.adj[v]:
                if not visited[u]:
                    visited[u] = True
                    self.depth[u] = self.depth[v] + 1
                    self.up[0][u] = v
                    q.append(u)

        # Fill sparse table: up[k][v] = 2^k-th ancestor of v
        for k in range(1, self.LOG):
            for v in range(self.n):
                anc = self.up[k - 1][v]
                self.up[k][v] = self.up[k - 1][anc]

    def kth_ancestor(self, v: int, k: int) -> int:
        """Return the k-th ancestor of v. Returns -1 if k > depth[v]."""
        if k > self.depth[v]:
            return -1
        for i in range(self.LOG):
            if (k >> i) & 1:
                v = self.up[i][v]
        return v

    def lca(self, u: int, v: int) -> int:
        """LCA of nodes u and v."""
        # Bring to same depth
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = self.depth[u] - self.depth[v]
        u = self.kth_ancestor(u, diff)
        if u == v:
            return u
        # Lift both simultaneously
        for k in range(self.LOG - 1, -1, -1):
            if self.up[k][u] != self.up[k][v]:
                u = self.up[k][u]
                v = self.up[k][v]
        return self.up[0][u]

    def distance(self, u: int, v: int) -> int:
        """Edge count between u and v."""
        l = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[l]


# ─────────────────────────────────────────────
# Method 2: Euler Tour + Sparse Table LCA (O(1) query)
# ─────────────────────────────────────────────
class LCAEulerTour:
    """
    O(1) LCA using Euler tour reduction to RMQ.
    first[v] = first occurrence of v in Euler tour.
    Then LCA(u,v) = node with minimum depth in euler[first[u]..first[v]].
    """

    def __init__(self, n: int, root: int = 0):
        self.n = n
        self.root = root
        self.adj: List[List[int]] = [[] for _ in range(n)]
        self.depth = [0] * n

    def add_edge(self, u: int, v: int) -> None:
        self.adj[u].append(v)
        self.adj[v].append(u)

    def build(self) -> None:
        """Build Euler tour and sparse table."""
        euler: List[int] = []
        first = [-1] * self.n
        depth = self.depth

        # Iterative DFS for Euler tour
        stack = [(self.root, -1, False)]
        while stack:
            v, parent, returning = stack.pop()
            if not returning:
                if first[v] == -1:
                    first[v] = len(euler)
                euler.append(v)
                stack.append((v, parent, True))
                for u in self.adj[v]:
                    if u != parent:
                        depth[u] = depth[v] + 1
                        stack.append((u, v, False))
            else:
                if parent != -1:
                    euler.append(parent)

        self.euler = euler
        self.first = first
        m = len(euler)
        LOG = max(1, m.bit_length())

        # Sparse table on (depth, index) for range min
        self._log = [0] * (m + 1)
        for i in range(2, m + 1):
            self._log[i] = self._log[i >> 1] + 1

        table = [[(depth[euler[i]], i) for i in range(m)]]
        for k in range(1, LOG):
            prev = table[k - 1]
            half = 1 << (k - 1)
            row = [min(prev[i], prev[i + half]) if i + half < m else prev[i]
                   for i in range(m)]
            table.append(row)

        self._table = table

    def _rmq(self, l: int, r: int) -> int:
        """Index of minimum-depth node in euler[l..r]."""
        k = self._log[r - l + 1]
        _, idx = min(self._table[k][l], self._table[k][r - (1 << k) + 1])
        return self.euler[idx]

    def lca(self, u: int, v: int) -> int:
        """O(1) LCA query."""
        l, r = self.first[u], self.first[v]
        if l > r:
            l, r = r, l
        return self._rmq(l, r)

    def distance(self, u: int, v: int) -> int:
        l = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[l]


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    #        0
    #       / \
    #      1   2
    #     / \   \
    #    3   4   5
    #           /
    #          6

    edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(5,6)]

    # Binary lifting
    bl = LCABinaryLifting(7, root=0)
    for u, v in edges:
        bl.add_edge(u, v)
    bl.build()

    assert bl.lca(3, 4) == 1
    assert bl.lca(3, 5) == 0
    assert bl.lca(6, 3) == 0
    assert bl.lca(6, 5) == 5
    assert bl.kth_ancestor(6, 2) == 2
    assert bl.distance(3, 6) == 5

    # Euler tour
    et = LCAEulerTour(7, root=0)
    for u, v in edges:
        et.add_edge(u, v)
    et.build()

    assert et.lca(3, 4) == 1
    assert et.lca(3, 5) == 0
    assert et.lca(6, 3) == 0
    assert et.lca(6, 5) == 5
    assert et.distance(3, 6) == 5

    print("All LCA tests passed.")
