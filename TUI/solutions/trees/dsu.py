"""
NAME: Disjoint Set Union (Union-Find) — Standard, Weighted, and Rollback DSU
TAGS: DSU, union-find, path compression, union by rank, weighted DSU, rollback, offline
DESCRIPTION: Standard DSU with path compression + union by rank achieves near O(1) amortized
    per operation. Weighted DSU tracks relative weights between nodes (useful for bipartite/parity
    problems). Rollback DSU supports undoing unions — required for offline dynamic connectivity
    and divide-and-conquer on edges.
COMPLEXITY: Standard: O(α(n)) per op; Rollback DSU: O(log n) per op (no path compression)
"""

from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# 1. Standard DSU — Path Compression + Union by Rank
# ─────────────────────────────────────────────
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.components = n

    def find(self, x: int) -> int:
        """Path compression (iterative)."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if they were in different components."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def get_size(self, x: int) -> int:
        return self.size[self.find(x)]


# ─────────────────────────────────────────────
# 2. Weighted DSU (Potential / Relative Weight DSU)
# Tracks weight[x] = weight of x relative to its root.
# Classic use: bipartite check, parity, distances in a group.
# ─────────────────────────────────────────────
class WeightedDSU:
    """
    weight[x] stores the "potential" of x relative to its root.
    For bipartite check: weight = parity (0 or 1).
    For distance problems: weight = distance to root.

    The combining rule: weight[x] = XOR/ADD/etc of edges along path to root.
    Default: XOR (bipartite parity).
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n   # weight[x] XOR root = parity to root

    def find(self, x: int) -> Tuple[int, int]:
        """Returns (root, weight_from_x_to_root) using path compression."""
        if self.parent[x] == x:
            return x, 0
        root, pw = self.find(self.parent[x])
        self.weight[x] ^= pw    # XOR-based; change ^ to + for additive weights
        self.parent[x] = root
        return root, self.weight[x]

    def union(self, x: int, y: int, w: int) -> bool:
        """
        Unite x and y with edge weight w (meaning: weight[y] XOR weight[x] = w).
        Returns False if already connected (and checks consistency).
        """
        rx, wx = self.find(x)
        ry, wy = self.find(y)
        if rx == ry:
            # Check consistency: wx XOR wy should equal w
            return (wx ^ wy) == w
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
            wx, wy = wy, wx
        self.parent[ry] = rx
        # weight[ry] XOR weight[rx] = wx XOR wy XOR w
        self.weight[ry] = wx ^ wy ^ w
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def diff(self, x: int, y: int) -> Optional[int]:
        """
        Returns weight difference between x and y if connected, else None.
        For bipartite: 0 = same side, 1 = different side.
        """
        rx, wx = self.find(x)
        ry, wy = self.find(y)
        if rx != ry:
            return None
        return wx ^ wy

    def is_bipartite_violation(self, x: int, y: int) -> bool:
        """Check if edge (x,y) would create an odd cycle (same side)."""
        d = self.diff(x, y)
        return d is not None and d == 0


# ─────────────────────────────────────────────
# 3. Rollback DSU (without path compression)
# Required for offline dynamic connectivity and
# divide & conquer on edges (e.g. Codeforces: add/remove edges, count components)
# ─────────────────────────────────────────────
class RollbackDSU:
    """
    DSU that supports undo of the last union.
    Uses union by rank but NO path compression (to make rollback possible).
    O(log n) per find/union/rollback.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.components = n
        self._history: List[Tuple] = []   # (x, px, rx_rank, y, py, ry_rank, components)

    def find(self, x: int) -> int:
        """Find without path compression."""
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Union and record history. Returns True if merged."""
        rx, ry = self.find(x), self.find(y)
        self._history.append((rx, self.parent[rx], self.rank[rx],
                               ry, self.parent[ry], self.rank[ry],
                               self.components))
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True

    def rollback(self) -> None:
        """Undo the last union operation."""
        if not self._history:
            return
        rx, prx, rrx, ry, pry, rry, comp = self._history.pop()
        self.parent[rx] = prx
        self.rank[rx] = rrx
        self.parent[ry] = pry
        self.rank[ry] = rry
        # Restore sizes
        if prx == rx and pry != ry:   # merge happened: ry was attached to rx
            self.size[rx] -= self.size[ry]
        elif pry == ry and prx != rx:
            self.size[ry] -= self.size[rx]
        self.components = comp

    def save(self) -> int:
        """Returns current stack depth (checkpoint)."""
        return len(self._history)

    def rollback_to(self, checkpoint: int) -> None:
        """Rollback to a saved checkpoint."""
        while len(self._history) > checkpoint:
            self.rollback()

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


# ─────────────────────────────────────────────
# 4. DSU on Tree (Small-to-Large / Dsu on tree)
# Used for subtree queries: count distinct colors, etc.
# ─────────────────────────────────────────────
def dsu_on_tree(n: int, adj: List[List[int]], vals: List[int]) -> List[int]:
    """
    DSU on tree (Sack): for each node, compute count of distinct values in its subtree.
    O(n log n) total using heavy child optimization.
    Returns answer[v] for each node v.
    """
    from collections import defaultdict

    size = [1] * n
    heavy = [-1] * n
    parent = [-1] * n

    # BFS to get order
    from collections import deque
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

    # Compute sizes and heavy children
    for v in reversed(order):
        best = 0
        for u in adj[v]:
            if u != parent[v]:
                size[v] += size[u]
                if size[u] > best:
                    best = size[u]
                    heavy[v] = u

    answer = [0] * n
    cnt: dict = defaultdict(int)
    distinct = [0]

    def add(v: int, p: int, sign: int) -> None:
        """Add/remove entire subtree of v (excluding edge to p)."""
        cnt[vals[v]] += sign
        if cnt[vals[v]] == 0:
            distinct[0] -= 1
        elif cnt[vals[v]] == 1 and sign == 1:
            distinct[0] += 1
        for u in adj[v]:
            if u != p:
                add(u, v, sign)

    def dfs(v: int, p: int, keep: bool) -> None:
        # Process light children
        for u in adj[v]:
            if u != p and u != heavy[v]:
                dfs(u, v, False)

        # Process heavy child (keep its data)
        if heavy[v] != -1:
            dfs(heavy[v], v, True)

        # Add all light children
        for u in adj[v]:
            if u != p and u != heavy[v]:
                add(u, v, 1)

        # Add v itself
        cnt[vals[v]] += 1
        if cnt[vals[v]] == 1:
            distinct[0] += 1

        answer[v] = distinct[0]

        # Clear if not keeping
        if not keep:
            add(v, p, -1)
            cnt[vals[v]] -= 1
            if cnt[vals[v]] == 0:
                distinct[0] -= 1

    import sys
    sys.setrecursionlimit(300000)
    dfs(0, -1, False)
    return answer


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Standard DSU
    dsu = DSU(5)
    dsu.union(0, 1)
    dsu.union(1, 2)
    assert dsu.connected(0, 2)
    assert not dsu.connected(0, 3)
    assert dsu.get_size(0) == 3
    assert dsu.components == 3

    # Weighted DSU (bipartite parity)
    wdsu = WeightedDSU(4)
    wdsu.union(0, 1, 1)   # 0 and 1 are on different sides
    wdsu.union(1, 2, 1)   # 1 and 2 are on different sides => 0 and 2 are same
    assert wdsu.diff(0, 2) == 0   # same side
    assert wdsu.diff(0, 1) == 1   # different sides
    # Adding 0-2 with w=1 would violate bipartite (they are same side)
    assert wdsu.is_bipartite_violation(0, 2)

    # Rollback DSU
    rdsu = RollbackDSU(4)
    rdsu.union(0, 1)
    rdsu.union(2, 3)
    assert rdsu.connected(0, 1)
    assert rdsu.connected(2, 3)
    checkpoint = rdsu.save()
    rdsu.union(0, 2)
    assert rdsu.connected(0, 3)
    rdsu.rollback_to(checkpoint)
    assert not rdsu.connected(0, 3)
    assert rdsu.connected(0, 1)   # previous unions still intact

    print("All DSU tests passed.")
