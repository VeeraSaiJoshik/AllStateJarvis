"""
NAME: Bipartite Check and Maximum Bipartite Matching (Hopcroft-Karp)
TAGS: graph, bipartite, matching, hopcroft-karp, 2-coloring, bfs, augmenting-path
DESCRIPTION: Checks if a graph is bipartite via 2-coloring (BFS/DFS), and computes
    maximum bipartite matching using Hopcroft-Karp (multiple augmenting paths per BFS
    phase). Use for assignment problems, maximum matching, vertex cover (König's theorem).
COMPLEXITY: Bipartite check O(V+E); Hopcroft-Karp O(E * sqrt(V)), Space O(V+E)
"""

from collections import defaultdict, deque
from typing import List, Tuple, Optional, Dict


# ──────────────────────────────────────────────
# 1. Bipartite Check via 2-Coloring (BFS)
# ──────────────────────────────────────────────

def is_bipartite(n: int, graph: defaultdict) -> Tuple[bool, List[int]]:
    """
    Checks if the graph is bipartite using BFS 2-coloring.
    Returns (is_bipartite, color[]) where color[v] ∈ {0, 1, -1}.
    color[v] = -1 means unvisited (isolated node — can be either side).
    Works on disconnected graphs.
    """
    color = [-1] * n

    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False, color   # odd cycle found

    return True, color


def get_bipartite_sets(n: int, graph: defaultdict
                        ) -> Optional[Tuple[List[int], List[int]]]:
    """
    Returns (left_set, right_set) if graph is bipartite, else None.
    """
    ok, color = is_bipartite(n, graph)
    if not ok:
        return None
    left  = [i for i in range(n) if color[i] == 0]
    right = [i for i in range(n) if color[i] == 1]
    return left, right


# ──────────────────────────────────────────────
# 2. Maximum Bipartite Matching — Hungarian (simple augmenting path)
#    O(V * E) — fine for small graphs (V ≤ 500).
# ──────────────────────────────────────────────

def max_bipartite_matching_hungarian(left_n: int, right_n: int,
                                     adj: List[List[int]]) -> int:
    """
    Maximum bipartite matching using augmenting paths (DFS per left node).
    adj[u] = list of right nodes that left node u connects to.
    Left nodes: 0..left_n-1, Right nodes: 0..right_n-1.
    Returns the size of the maximum matching.
    """
    match_left  = [-1] * left_n    # match_left[u] = right node matched to u (-1 = free)
    match_right = [-1] * right_n   # match_right[v] = left node matched to v (-1 = free)

    def _augment(u: int, visited: List[bool]) -> bool:
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_right[v] == -1 or _augment(match_right[v], visited):
                    match_left[u] = v
                    match_right[v] = u
                    return True
        return False

    matching = 0
    for u in range(left_n):
        visited = [False] * right_n
        if _augment(u, visited):
            matching += 1

    return matching


# ──────────────────────────────────────────────
# 3. Hopcroft-Karp Maximum Bipartite Matching
#    O(E * sqrt(V)) — handles large graphs efficiently.
#    BFS to find shortest augmenting paths, then DFS to augment all simultaneously.
# ──────────────────────────────────────────────

INF = float('inf')

class HopcroftKarp:
    """
    Maximum bipartite matching via Hopcroft-Karp.
    Left nodes: 0..left_n-1 (use as-is).
    Right nodes: 0..right_n-1 (internally offset — use through the API).
    """

    def __init__(self, left_n: int, right_n: int):
        self.left_n = left_n
        self.right_n = right_n
        self.adj: List[List[int]] = [[] for _ in range(left_n)]

    def add_edge(self, u: int, v: int):
        """Add edge between left node u and right node v."""
        self.adj[u].append(v)

    def _bfs(self) -> bool:
        """BFS to build layered graph. Returns True if augmenting path exists."""
        self.dist_u = [INF] * self.left_n
        queue: deque = deque()
        for u in range(self.left_n):
            if self.match_u[u] == -1:
                self.dist_u[u] = 0
                queue.append(u)
        found = False
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                w = self.match_v[v]
                if w == -1:
                    found = True
                elif self.dist_u[w] == INF:
                    self.dist_u[w] = self.dist_u[u] + 1
                    queue.append(w)
        return found

    def _dfs(self, u: int) -> bool:
        """DFS augmentation along the layered graph."""
        for v in self.adj[u]:
            w = self.match_v[v]
            if w == -1 or (self.dist_u[w] == self.dist_u[u] + 1 and self._dfs(w)):
                self.match_u[u] = v
                self.match_v[v] = u
                return True
        self.dist_u[u] = INF    # remove from layered graph
        return False

    def max_matching(self) -> int:
        """Returns the size of the maximum matching."""
        self.match_u = [-1] * self.left_n    # match_u[u] = right node
        self.match_v = [-1] * self.right_n   # match_v[v] = left node
        matching = 0
        while self._bfs():
            for u in range(self.left_n):
                if self.match_u[u] == -1:
                    if self._dfs(u):
                        matching += 1
        return matching

    def get_matching(self) -> List[Tuple[int, int]]:
        """Returns list of (left_node, right_node) pairs in the matching."""
        return [(u, self.match_u[u]) for u in range(self.left_n)
                if self.match_u[u] != -1]

    def min_vertex_cover(self) -> Tuple[List[int], List[int]]:
        """
        Returns minimum vertex cover (König's theorem):
        size of min vertex cover = size of max matching.
        Returns (left_cover, right_cover).
        """
        self.max_matching()
        # BFS from unmatched left nodes through alternating paths
        reachable_left = set()
        reachable_right = set()
        queue: deque = deque()

        for u in range(self.left_n):
            if self.match_u[u] == -1:
                queue.append(u)
                reachable_left.add(u)

        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if v not in reachable_right:
                    reachable_right.add(v)
                    w = self.match_v[v]
                    if w != -1 and w not in reachable_left:
                        reachable_left.add(w)
                        queue.append(w)

        left_cover  = [u for u in range(self.left_n) if u not in reachable_left]
        right_cover = [v for v in range(self.right_n) if v in reachable_right]
        return left_cover, right_cover


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Bipartite check: cycle of length 4 (bipartite)
    g: defaultdict = defaultdict(list)
    for u, v in [(0,1),(1,2),(2,3),(3,0)]:
        g[u].append(v)
        g[v].append(u)
    ok, color = is_bipartite(4, g)
    print("Is bipartite:", ok)                # True
    print("Colors:", color)                   # [0,1,0,1] or similar

    # Odd cycle: not bipartite
    g2: defaultdict = defaultdict(list)
    for u, v in [(0,1),(1,2),(2,0)]:
        g2[u].append(v)
        g2[v].append(u)
    print("Triangle bipartite:", is_bipartite(3, g2)[0])  # False

    # Hopcroft-Karp matching
    # Left: {0,1,2}, Right: {0,1,2}
    # Edges: 0-0, 0-1, 1-1, 1-2, 2-2
    hk = HopcroftKarp(3, 3)
    for u, v in [(0,0),(0,1),(1,1),(1,2),(2,2)]:
        hk.add_edge(u, v)
    print("Max matching (HK):", hk.max_matching())   # 3
    print("Matching pairs:", hk.get_matching())       # [(0,0),(1,1),(2,2)] or similar

    # Hungarian (simple)
    adj = [[0, 1], [1, 2], [2]]   # left 0 -> right {0,1}, left 1 -> {1,2}, left 2 -> {2}
    print("Max matching (Hungarian):", max_bipartite_matching_hungarian(3, 3, adj))  # 3

    # Min vertex cover
    lc, rc = hk.min_vertex_cover()
    print("Min vertex cover — left:", lc, "right:", rc)
