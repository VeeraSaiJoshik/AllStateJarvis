"""
NAME: Kosaraju's Strongly Connected Components (SCC)
TAGS: graph, scc, strongly-connected, kosaraju, directed, dfs
DESCRIPTION: Finds all strongly connected components (SCCs) in a directed graph using
    two DFS passes: one on the original graph (to get finish order) and one on the
    transposed graph. Use for condensation DAGs, 2-SAT, and reachability problems.
COMPLEXITY: Time O(V + E), Space O(V + E)
"""

from collections import defaultdict
from typing import List, Tuple


# ──────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────

def build_graph(n: int, edges: List[Tuple[int, int]]) -> defaultdict:
    g: defaultdict = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
    return g


def transpose_graph(n: int, edges: List[Tuple[int, int]]) -> defaultdict:
    """Returns the graph with all edge directions reversed."""
    g: defaultdict = defaultdict(list)
    for u, v in edges:
        g[v].append(u)
    return g


# ──────────────────────────────────────────────
# 1. Kosaraju's Algorithm — SCC
# ──────────────────────────────────────────────

def kosaraju_scc(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Returns list of SCCs. Each SCC is a list of node indices (0-indexed).
    The SCCs are returned in reverse topological order of the condensation DAG
    (i.e., the SCC reachable from everything is first).
    """
    g = build_graph(n, edges)
    gt = transpose_graph(n, edges)

    # ── Pass 1: DFS on original graph, record finish order ──
    visited = [False] * n
    finish_order: List[int] = []

    def dfs1(u: int):
        """Iterative DFS to avoid recursion limit."""
        stack = [(u, False)]
        while stack:
            node, returning = stack.pop()
            if returning:
                finish_order.append(node)
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))          # push return marker
            for v in g[node]:
                if not visited[v]:
                    stack.append((v, False))

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    # ── Pass 2: DFS on transposed graph in reverse finish order ──
    visited2 = [False] * n
    sccs: List[List[int]] = []

    def dfs2(u: int) -> List[int]:
        """Collect all nodes in the same SCC on transposed graph."""
        comp = []
        stack = [u]
        while stack:
            node = stack.pop()
            if visited2[node]:
                continue
            visited2[node] = True
            comp.append(node)
            for v in gt[node]:
                if not visited2[v]:
                    stack.append(v)
        return comp

    for u in reversed(finish_order):
        if not visited2[u]:
            comp = dfs2(u)
            sccs.append(comp)

    return sccs


# ──────────────────────────────────────────────
# 2. SCC Component ID array
#    comp_id[v] = which SCC node v belongs to (0-indexed).
# ──────────────────────────────────────────────

def scc_ids(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    """
    Returns (num_sccs, comp_id) where comp_id[v] = SCC index of node v.
    SCC indices are in topological order of the condensation DAG
    (comp 0 has no incoming edges from other SCCs).
    """
    sccs = kosaraju_scc(n, edges)
    comp_id = [-1] * n
    # kosaraju returns SCCs in reverse topo order; reverse to get topo order
    for idx, comp in enumerate(reversed(sccs)):
        for v in comp:
            comp_id[v] = idx
    return len(sccs), comp_id


# ──────────────────────────────────────────────
# 3. Condensation DAG
#    Contract each SCC to a single super-node.
#    Useful for solving problems on the DAG of SCCs.
# ──────────────────────────────────────────────

def condensation_dag(n: int, edges: List[Tuple[int, int]]
                     ) -> Tuple[int, List[Tuple[int, int]], List[int]]:
    """
    Returns (num_sccs, dag_edges, comp_id).
    dag_edges: unique directed edges between distinct SCCs.
    comp_id[v] = SCC index of node v (topological order).
    """
    num_sccs, comp_id = scc_ids(n, edges)
    dag_edge_set: set = set()
    for u, v in edges:
        cu, cv = comp_id[u], comp_id[v]
        if cu != cv:
            dag_edge_set.add((cu, cv))
    return num_sccs, list(dag_edge_set), comp_id


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Graph with SCCs: {0,1,2}, {3}, {4}
    # 0->1->2->0 (cycle), 1->3, 3->4
    edges = [(0,1),(1,2),(2,0),(1,3),(3,4)]
    n = 5

    sccs = kosaraju_scc(n, edges)
    print("SCCs:")
    for scc in sccs:
        print(" ", sorted(scc))
    # [{0,1,2}, {3}, {4}] in some order

    num, comp_id = scc_ids(n, edges)
    print(f"Num SCCs: {num}")           # 3
    print("Component IDs:", comp_id)    # e.g. [0,0,0,1,2]

    num_dag, dag_edges, comp_id = condensation_dag(n, edges)
    print(f"Condensation DAG ({num_dag} nodes):", dag_edges)

    # Verify: graph where every node is its own SCC
    edges2 = [(0,1),(1,2),(2,3)]
    sccs2 = kosaraju_scc(4, edges2)
    print("Linear graph SCCs:", [sorted(s) for s in sccs2])   # [[0],[1],[2],[3]]
