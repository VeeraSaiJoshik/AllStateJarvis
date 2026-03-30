"""
NAME: Tarjan's SCC, Bridges, and Articulation Points
TAGS: graph, scc, bridges, articulation-points, tarjan, dfs, low-link
DESCRIPTION: Single-pass DFS algorithm using low-link values to find SCCs, bridges
    (edges whose removal disconnects the graph), and articulation points (vertices whose
    removal disconnects the graph). Essential for network reliability and 2-edge connectivity.
COMPLEXITY: Time O(V + E), Space O(V)
"""

from collections import defaultdict
from typing import List, Set, Tuple


# ──────────────────────────────────────────────
# 1. Tarjan's SCC (directed graph)
#    low[u] = min discovery time reachable from u's subtree.
#    Node u is SCC root iff disc[u] == low[u].
# ──────────────────────────────────────────────

def tarjan_scc(n: int, graph: defaultdict) -> List[List[int]]:
    """
    Finds all SCCs in a directed graph using Tarjan's algorithm.
    graph[u] = list of neighbors v (directed edges u->v).
    Returns list of SCCs; each SCC is a list of node indices (0-indexed).
    SCCs are returned in reverse topological order.
    """
    disc = [-1] * n          # discovery time (-1 = unvisited)
    low = [0] * n            # low-link value
    on_stack = [False] * n
    stack: List[int] = []
    sccs: List[List[int]] = []
    timer = [0]

    def _dfs(u: int):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        stack.append(u)
        on_stack[u] = True

        for v in graph[u]:
            if disc[v] == -1:
                _dfs(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:
                low[u] = min(low[u], disc[v])

        # u is the root of an SCC
        if low[u] == disc[u]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == u:
                    break
            sccs.append(comp)

    for i in range(n):
        if disc[i] == -1:
            _dfs(i)

    return sccs


# ──────────────────────────────────────────────
# Iterative Tarjan's SCC (avoids Python recursion limit)
# ──────────────────────────────────────────────

def tarjan_scc_iterative(n: int, graph: defaultdict) -> List[List[int]]:
    """
    Iterative version of Tarjan's SCC — safe for large n.
    """
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    tarjan_stack: List[int] = []
    sccs: List[List[int]] = []
    timer = 0

    # Explicit call stack: (node, iterator_over_neighbors, index_in_neighbors)
    for start in range(n):
        if disc[start] != -1:
            continue
        call_stack = [(start, iter(graph[start]))]
        disc[start] = low[start] = timer
        timer += 1
        tarjan_stack.append(start)
        on_stack[start] = True

        while call_stack:
            u, neighbors = call_stack[-1]
            try:
                v = next(neighbors)
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    tarjan_stack.append(v)
                    on_stack[v] = True
                    call_stack.append((v, iter(graph[v])))
                elif on_stack[v]:
                    low[u] = min(low[u], disc[v])
            except StopIteration:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    low[parent] = min(low[parent], low[u])
                if low[u] == disc[u]:
                    comp = []
                    while True:
                        w = tarjan_stack.pop()
                        on_stack[w] = False
                        comp.append(w)
                        if w == u:
                            break
                    sccs.append(comp)

    return sccs


# ──────────────────────────────────────────────
# 2. Bridges (undirected graph)
#    An edge (u, v) is a bridge if removing it disconnects the graph.
#    Bridge iff low[v] > disc[u] (v cannot reach u or above without the edge).
# ──────────────────────────────────────────────

def find_bridges(n: int, graph: defaultdict) -> List[Tuple[int, int]]:
    """
    Finds all bridges in an undirected graph.
    Returns list of (u, v) bridge edges (u < v).
    graph[u] = list of neighbors.
    """
    disc = [-1] * n
    low = [0] * n
    bridges: List[Tuple[int, int]] = []
    timer = [0]

    def _dfs(u: int, parent: int):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in graph[u]:
            if disc[v] == -1:
                _dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((min(u, v), max(u, v)))
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            _dfs(i, -1)

    return bridges


# ──────────────────────────────────────────────
# 3. Articulation Points (undirected graph)
#    A vertex u is an articulation point if:
#    - u is root of DFS tree and has ≥ 2 children, OR
#    - u is not root and has child v where low[v] >= disc[u].
# ──────────────────────────────────────────────

def find_articulation_points(n: int, graph: defaultdict) -> List[int]:
    """
    Finds all articulation points (cut vertices) in an undirected graph.
    Returns sorted list of articulation point node indices.
    """
    disc = [-1] * n
    low = [0] * n
    is_ap = [False] * n
    timer = [0]

    def _dfs(u: int, parent: int):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for v in graph[u]:
            if disc[v] == -1:
                children += 1
                _dfs(v, u)
                low[u] = min(low[u], low[v])
                # Non-root: child v can't reach above u
                if parent != -1 and low[v] >= disc[u]:
                    is_ap[u] = True
                # Root with multiple DFS children
                if parent == -1 and children > 1:
                    is_ap[u] = True
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            _dfs(i, -1)

    return [i for i in range(n) if is_ap[i]]


# ──────────────────────────────────────────────
# 4. Bridges + Articulation Points — combined pass
# ──────────────────────────────────────────────

def find_bridges_and_aps(n: int, graph: defaultdict
                          ) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Single DFS pass to find both bridges and articulation points.
    Returns (bridges, articulation_points).
    """
    disc = [-1] * n
    low = [0] * n
    is_ap = [False] * n
    bridges: List[Tuple[int, int]] = []
    timer = [0]

    def _dfs(u: int, parent: int):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for v in graph[u]:
            if disc[v] == -1:
                children += 1
                _dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((min(u, v), max(u, v)))
                if parent != -1 and low[v] >= disc[u]:
                    is_ap[u] = True
                if parent == -1 and children > 1:
                    is_ap[u] = True
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            _dfs(i, -1)

    return bridges, [i for i in range(n) if is_ap[i]]


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from collections import defaultdict

    # Directed graph for SCC: 0->1->2->0, 1->3, 3->4->3 (cycle)
    dg: defaultdict = defaultdict(list)
    for u, v in [(0,1),(1,2),(2,0),(1,3),(3,4),(4,3)]:
        dg[u].append(v)

    sccs = tarjan_scc(5, dg)
    print("Tarjan SCCs:")
    for scc in sccs:
        print(" ", sorted(scc))
    # [{3,4}, {0,1,2}] or similar

    sccs_it = tarjan_scc_iterative(5, dg)
    print("Iterative SCCs:", [sorted(s) for s in sccs_it])

    # Undirected graph for bridges/APs
    # 0-1-2-3, also 1-3 (so 0-1 is a bridge, 0 and 2,3 via 1)
    ug: defaultdict = defaultdict(list)
    for u, v in [(0,1),(1,2),(2,3),(1,3)]:
        ug[u].append(v)
        ug[v].append(u)

    bridges = find_bridges(4, ug)
    print("Bridges:", bridges)   # [(0,1)]

    aps = find_articulation_points(4, ug)
    print("Articulation points:", aps)   # [1]

    b2, ap2 = find_bridges_and_aps(4, ug)
    print("Combined — bridges:", b2, "| APs:", ap2)
