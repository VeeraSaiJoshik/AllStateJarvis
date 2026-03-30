"""
NAME: Cycle Detection (Directed and Undirected Graphs)
TAGS: graph, cycle, dfs, directed, undirected, floyd, tortoise-hare, topological
DESCRIPTION: Detects cycles in directed graphs using DFS with 3-color marking (white/
    gray/black) and in undirected graphs using parent-tracking DFS or DSU. Also includes
    Floyd's cycle detection for linked-list-style sequences. Use in dependency validation,
    deadlock detection, and DAG verification.
COMPLEXITY: DFS-based O(V + E), Space O(V); Floyd's O(N) time O(1) space
"""

from collections import defaultdict, deque
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────
# 1. Cycle Detection in Directed Graph — DFS (3-color)
#    WHITE=0, GRAY=1 (in recursion stack), BLACK=2 (done)
#    Back edge (node in GRAY) = cycle.
# ──────────────────────────────────────────────

def has_cycle_directed_dfs(n: int, graph: defaultdict) -> bool:
    """
    Returns True if the directed graph (0-indexed, n nodes) contains a cycle.
    """
    color = [0] * n   # 0=white, 1=gray, 2=black

    def _dfs(u: int) -> bool:
        color[u] = 1
        for v in graph[u]:
            if color[v] == 1:    # back edge → cycle
                return True
            if color[v] == 0 and _dfs(v):
                return True
        color[u] = 2
        return False

    return any(_dfs(i) for i in range(n) if color[i] == 0)


def find_cycle_directed(n: int, graph: defaultdict) -> Optional[List[int]]:
    """
    Returns one cycle as a list of nodes (in order) if it exists, else None.
    """
    color = [0] * n
    parent = [-1] * n
    cycle_start = [-1]

    def _dfs(u: int) -> bool:
        color[u] = 1
        for v in graph[u]:
            if color[v] == 1:
                cycle_start[0] = v
                parent[v] = u    # override to close the loop
                return True
            if color[v] == 0:
                parent[v] = u
                if _dfs(v):
                    return True
        color[u] = 2
        return False

    for i in range(n):
        if color[i] == 0:
            if _dfs(i):
                # Reconstruct cycle starting from cycle_start
                start = cycle_start[0]
                cycle = []
                node = start
                # Walk backwards via parent until we reach start again
                # Easier: just record the cycle by traversal
                seen = {}
                cur = start
                while cur not in seen:
                    seen[cur] = len(cycle)
                    cycle.append(cur)
                    cur = parent[cur]
                # cur is now the repeated node; slice from its first occurrence
                idx = seen[cur]
                return cycle[idx:] + [cur]
    return None


# ──────────────────────────────────────────────
# 2. Cycle Detection in Directed Graph — Kahn's (Topological Sort)
#    If topological sort doesn't include all nodes → cycle exists.
# ──────────────────────────────────────────────

def has_cycle_directed_kahn(n: int, graph: defaultdict) -> bool:
    """
    Detects cycle in a directed graph using Kahn's algorithm.
    Returns True if a cycle exists.
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1

    queue: deque = deque(i for i in range(n) if in_degree[i] == 0)
    visited_count = 0

    while queue:
        u = queue.popleft()
        visited_count += 1
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return visited_count != n   # True if not all nodes processed → cycle


# ──────────────────────────────────────────────
# 3. Cycle Detection in Undirected Graph — DFS with parent tracking
# ──────────────────────────────────────────────

def has_cycle_undirected_dfs(n: int, graph: defaultdict) -> bool:
    """
    Returns True if the undirected graph has a cycle.
    Uses DFS with parent tracking. Handles disconnected graphs.
    """
    visited = [False] * n

    def _dfs(u: int, parent: int) -> bool:
        visited[u] = True
        for v in graph[u]:
            if not visited[v]:
                if _dfs(v, u):
                    return True
            elif v != parent:    # back edge (not the tree edge back to parent)
                return True
        return False

    return any(_dfs(i, -1) for i in range(n) if not visited[i])


def has_cycle_undirected_dfs_multigraph(n: int, graph: defaultdict) -> bool:
    """
    Cycle detection for multigraphs (multiple edges between same nodes).
    Track parent by edge index to handle parallel edges correctly.
    graph[u] = list of (v, edge_id).
    """
    visited = [False] * n

    def _dfs(u: int, parent_edge: int) -> bool:
        visited[u] = True
        for v, eid in graph[u]:
            if not visited[v]:
                if _dfs(v, eid):
                    return True
            elif eid != parent_edge:
                return True
        return False

    return any(_dfs(i, -1) for i in range(n) if not visited[i])


# ──────────────────────────────────────────────
# 4. Cycle Detection in Undirected Graph — DSU (Union-Find)
#    If we try to union two nodes already in the same component → cycle.
# ──────────────────────────────────────────────

def has_cycle_undirected_dsu(n: int, edges: List[Tuple[int, int]]) -> bool:
    """
    Detects cycle in an undirected graph using Union-Find.
    edges: list of (u, v).
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False   # already connected → cycle
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    for u, v in edges:
        if not union(u, v):
            return True
    return False


# ──────────────────────────────────────────────
# 5. Floyd's Cycle Detection (Tortoise and Hare)
#    For functional graphs / linked-list-style sequences.
#    f: int -> int (next element function)
# ──────────────────────────────────────────────

def floyd_cycle_detection(f, start: int) -> Tuple[int, int]:
    """
    Floyd's tortoise and hare algorithm.
    f(x) maps a value to the next value in the sequence.
    Returns (mu, lam) where:
      mu  = start of the cycle (index of first repeated element)
      lam = length of the cycle.
    """
    # Phase 1: Find meeting point inside the cycle
    slow = f(start)
    fast = f(f(start))
    while slow != fast:
        slow = f(slow)
        fast = f(f(fast))

    # Phase 2: Find start of cycle (mu)
    mu = 0
    slow = start
    while slow != fast:
        slow = f(slow)
        fast = f(fast)
        mu += 1

    # Phase 3: Find cycle length (lambda)
    lam = 1
    fast = f(slow)
    while slow != fast:
        fast = f(fast)
        lam += 1

    return mu, lam


# ──────────────────────────────────────────────
# 6. Iterative DFS cycle detection (safe for large graphs)
# ──────────────────────────────────────────────

def has_cycle_directed_iterative(n: int, graph: defaultdict) -> bool:
    """
    Iterative version of directed cycle detection (avoids recursion limit).
    """
    color = [0] * n   # 0=white, 1=gray, 2=black

    for start in range(n):
        if color[start] != 0:
            continue
        # Stack stores (node, returning?)
        stack = [(start, False)]
        while stack:
            u, returning = stack.pop()
            if returning:
                color[u] = 2
                continue
            if color[u] == 1:
                return True
            if color[u] == 2:
                continue
            color[u] = 1
            stack.append((u, True))   # mark to blacken when done
            for v in graph[u]:
                if color[v] == 1:
                    return True
                if color[v] == 0:
                    stack.append((v, False))

    return False


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Directed graph with cycle: 0->1->2->0
    dg: defaultdict = defaultdict(list)
    for u, v in [(0,1),(1,2),(2,0),(3,4)]:
        dg[u].append(v)

    print("Directed cycle (DFS):", has_cycle_directed_dfs(5, dg))        # True
    print("Directed cycle (Kahn):", has_cycle_directed_kahn(5, dg))      # True
    print("Directed cycle (iter):", has_cycle_directed_iterative(5, dg)) # True
    print("Cycle nodes:", find_cycle_directed(5, dg))                    # [0,1,2] or similar

    # Directed DAG (no cycle): 0->1->3, 0->2->3
    dag: defaultdict = defaultdict(list)
    for u, v in [(0,1),(0,2),(1,3),(2,3)]:
        dag[u].append(v)
    print("DAG has cycle:", has_cycle_directed_dfs(4, dag))              # False

    # Undirected graph with cycle: 0-1-2-0
    ug: defaultdict = defaultdict(list)
    for u, v in [(0,1),(1,2),(2,0)]:
        ug[u].append(v)
        ug[v].append(u)
    print("Undirected cycle (DFS):", has_cycle_undirected_dfs(3, ug))   # True
    print("Undirected cycle (DSU):", has_cycle_undirected_dsu(3, [(0,1),(1,2),(2,0)]))  # True

    # Floyd's: f(x) = (x * x + 1) % 11, start=3 → known cycle
    f = lambda x: (x * x + 1) % 11
    mu, lam = floyd_cycle_detection(f, 3)
    print(f"Floyd's: cycle start={mu}, cycle length={lam}")
