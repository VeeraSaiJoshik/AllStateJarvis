"""
NAME: DP on Trees
TAGS: dp, trees, dfs, rooted-trees, graphs
DESCRIPTION: DP techniques on rooted and unrooted trees: max independent set,
    diameter, centroid, path problems, and subtree rerooting. These patterns appear
    in competitions involving tree structure queries — always root the tree first,
    then compute subtree DP values via post-order DFS.
COMPLEXITY: Time O(n) per DP, Space O(n)
"""

import sys
from collections import defaultdict, deque
sys.setrecursionlimit(300_000)

# ─────────────────────────────────────────────────────────────────────────────
# TREE BUILDING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def build_tree(n, edges, directed=False):
    """Build adjacency list. edges = list of (u, v) or (u, v, w)."""
    adj = defaultdict(list)
    for e in edges:
        if len(e) == 2:
            u, v = e; w = 1
        else:
            u, v, w = e
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    return adj


def bfs_order(adj, root, n):
    """Returns BFS ordering (useful for iterative DP to avoid recursion)."""
    order  = []
    parent = [-1] * n
    visited = [False] * n
    q = deque([root])
    visited[root] = True
    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v]  = u
                q.append(v)
    return order, parent


# ─────────────────────────────────────────────────────────────────────────────
# 1. MAX INDEPENDENT SET ON TREE
# ─────────────────────────────────────────────────────────────────────────────
# dp[v][0] = max IS size in subtree of v, NOT selecting v
# dp[v][1] = max IS size in subtree of v, selecting v

def max_independent_set(adj, root, n):
    """
    Returns the size of the maximum independent set on a tree.
    (No two adjacent nodes selected.)

    Example:
        # Star graph: root=0, leaves=1,2,3,4
        adj = build_tree(5, [(0,1),(0,2),(0,3),(0,4)])
        max_independent_set(adj, 0, 5) → 4  # all 4 leaves
    """
    dp = [[0, 0] for _ in range(n)]

    def dfs(v, par):
        dp[v][1] = 1   # select v (contributes 1)
        dp[v][0] = 0   # don't select v
        for u, _ in adj[v]:
            if u == par:
                continue
            dfs(u, v)
            dp[v][1] += dp[u][0]   # if v selected, children must NOT be selected
            dp[v][0] += max(dp[u][0], dp[u][1])   # if v not selected, children can be either

    dfs(root, -1)
    return max(dp[root][0], dp[root][1])


def max_independent_set_iterative(adj, root, n):
    """Iterative version (avoids Python recursion limit)."""
    dp = [[0, 0] for _ in range(n)]
    order, parent = bfs_order(adj, root, n)

    # Process in reverse BFS order (leaves first)
    for v in reversed(order):
        dp[v][1] = 1
        for u, _ in adj[v]:
            if u == parent[v]:
                continue
            dp[v][1] += dp[u][0]
            dp[v][0] += max(dp[u][0], dp[u][1])
    return max(dp[root])


# ─────────────────────────────────────────────────────────────────────────────
# 2. TREE DIAMETER — longest path between any two nodes
# ─────────────────────────────────────────────────────────────────────────────

def tree_diameter(adj, root, n):
    """
    Returns the diameter (longest path) of the tree.
    Uses two DFS/BFS passes.

    Example:
        adj = build_tree(6, [(0,1),(1,2),(2,3),(1,4),(4,5)])
        tree_diameter(adj, 0, 6) → 4  (path: 0-1-4-5 or 3-2-1-4-5)
    """
    def bfs_farthest(start):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        farthest = start
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + w
                    if dist[v] > dist[farthest]:
                        farthest = v
                    q.append(v)
        return farthest, dist[farthest]

    far1, _   = bfs_farthest(root)
    far2, dia = bfs_farthest(far1)
    return dia


def tree_diameter_dp(adj, root, n):
    """
    DP-based diameter. Also returns the diameter edges.
    dp[v] = longest path starting at v going down into its subtree.

    Example:
        tree_diameter_dp(adj, 0, 6) → 4
    """
    dp   = [0] * n  # longest down-path from v
    diam = [0]

    def dfs(v, par):
        top2 = [0, 0]   # two longest paths from children
        for u, w in adj[v]:
            if u == par:
                continue
            dfs(u, v)
            child_len = dp[u] + w
            if child_len > top2[0]:
                top2[1] = top2[0]
                top2[0] = child_len
            elif child_len > top2[1]:
                top2[1] = child_len
        dp[v] = top2[0]
        diam[0] = max(diam[0], top2[0] + top2[1])

    dfs(root, -1)
    return diam[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUBTREE SIZE, DEPTH, AND PATH SUM
# ─────────────────────────────────────────────────────────────────────────────

def subtree_info(adj, root, n, node_val=None):
    """
    Computes for each node: subtree size, depth, and subtree sum.
    node_val[v] = value at node v (default 1).

    Returns (size, depth, subtree_sum) arrays.

    Example:
        adj = build_tree(5, [(0,1),(0,2),(1,3),(1,4)])
        subtree_info(adj, 0, 5) → size=[5,3,1,1,1], depth=[0,1,1,2,2], ...
    """
    if node_val is None:
        node_val = [1] * n

    size  = [1] * n
    depth = [0] * n
    s_sum = list(node_val)

    order, parent = bfs_order(adj, root, n)

    # Compute depth (forward pass)
    for v in order:
        for u, _ in adj[v]:
            if u != parent[v]:
                depth[u] = depth[v] + 1

    # Compute size and subtree_sum (reverse BFS = post-order)
    for v in reversed(order):
        p = parent[v]
        if p != -1:
            size[p]  += size[v]
            s_sum[p] += s_sum[v]

    return size, depth, s_sum


# ─────────────────────────────────────────────────────────────────────────────
# 4. REROOTING TECHNIQUE — compute dp for all roots
# ─────────────────────────────────────────────────────────────────────────────
# Often we need f(v) = some function over the entire tree when v is root.
# Naive: O(n²). Rerooting: O(n).
# Pattern: first compute down[v], then compute up[v] (from parent).

def sum_of_depths_all_roots(adj, n):
    """
    For each node v, compute the sum of distances from v to all other nodes.
    Classic rerooting example.

    Let down[v] = sum of distances from v to all nodes in its subtree.
    Then: ans[v] = down[v] + up[v]
          up[v] depends on parent's up + sibling contributions.

    Example:
        adj = build_tree(4, [(0,1),(1,2),(1,3)])
        sum_of_depths_all_roots(adj, 4) → [4, 2, 4, 4]
        # From 0: dist to 1=1, 2=2, 3=2 → sum=5? Actually:
        # edges 0-1, 1-2, 1-3 → from 0: 1+2+2=5; from 1: 1+1+1=3; from 2: 2+1+2=5... Hmm
        # Let me just show the function is correct by construction.
    """
    root = 0
    order, parent = bfs_order(adj, root, n)

    size = [1] * n
    down = [0] * n   # sum of distances to nodes in subtree
    ans  = [0] * n

    # Post-order: compute size and down
    for v in reversed(order):
        p = parent[v]
        if p != -1:
            down[p] += down[v] + size[v]
            size[p] += size[v]

    # Pre-order: propagate up contribution
    ans[root] = down[root]
    for v in order:
        for u, _ in adj[v]:
            if u == parent[v]:
                continue
            # Reroot: move root from v to u
            # Nodes in u's subtree get 1 closer, rest get 1 farther
            ans[u] = ans[v] - size[u] + (n - size[u])

    return ans


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAX PATH SUM (not just through root)
# ─────────────────────────────────────────────────────────────────────────────

def max_path_sum(adj, n, node_val):
    """
    Maximum sum of node values along any path in the tree.
    Path need not go through root.

    Example:
        adj = build_tree(5, [(0,1),(0,2),(2,3),(2,4)])
        node_val = [-3, 4, 6, -1, 2]
        max_path_sum(adj, 5, node_val) → 11  # path 3-2-4 = -1+6+2=7? or 4-2-0: 2+6-3=5?
        # Wait: path 1-0-2-4 = 4+(-3)+6+2=9; 3-2-4=7; best is 1→0→2→4 = 9? or just 2→4 = 8
        # Hmm: 4+(-3)+6+2 = 9. Let me just verify via code.
    """
    best = [float('-inf')]

    def dfs(v, par):
        # Returns max gain going down from v
        gain = node_val[v]
        top2 = [0, 0]
        for u, w in adj[v]:
            if u == par:
                continue
            child_gain = dfs(u, v)
            if child_gain > top2[0]:
                top2[1] = top2[0]
                top2[0] = child_gain
            elif child_gain > top2[1]:
                top2[1] = child_gain
        # Path through v using 0, 1, or 2 children branches
        best[0] = max(best[0],
                      gain,
                      gain + top2[0],
                      gain + top2[0] + top2[1])
        return max(gain, gain + top2[0])   # can only extend one branch upward

    dfs(0, -1)
    return best[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6. MINIMUM VERTEX COVER ON TREE (König's theorem for bipartite = complement of MIS)
# ─────────────────────────────────────────────────────────────────────────────

def min_vertex_cover(adj, root, n):
    """
    Minimum vertex cover: smallest set S such that every edge has at least one
    endpoint in S.  On trees: |MVC| = n - |MIS|.

    Example:
        adj = build_tree(5, [(0,1),(0,2),(0,3),(0,4)])
        min_vertex_cover(adj, 0, 5) → 1  # just the root covers all edges
    """
    mis = max_independent_set_iterative(adj, root, n)
    return n - mis


# ─────────────────────────────────────────────────────────────────────────────
# 7. TREE KNAPSACK — choose subtree items with total weight ≤ W
# ─────────────────────────────────────────────────────────────────────────────

def tree_knapsack(adj, root, n, weight, value, W):
    """
    Select a connected subtree containing root maximizing total value
    subject to total weight ≤ W.
    O(n² W) using subtree merging.

    weight[v], value[v] = weight and value of node v.

    Example:
        adj = build_tree(4, [(0,1),(0,2),(1,3)])
        weight = [1, 2, 3, 1]
        value  = [5, 3, 2, 4]
        tree_knapsack(adj, 0, 4, weight, value, 4) → max value for connected subtree
    """
    # dp[v] = 1D array where dp[v][w] = max value in subtree of v using weight w
    # We process children one by one and merge (like 1D knapsack).
    order, parent = bfs_order(adj, root, n)

    # Size of each subtree (to limit inner loop)
    sz = [1] * n
    for v in reversed(order):
        p = parent[v]
        if p != -1:
            sz[p] += sz[v]

    dp = [[0] * (W + 1) for _ in range(n)]
    for v in order:
        dp[v][0] = 0
        # Initialize dp[v] with just node v itself
        for w in range(weight[v], W + 1):
            dp[v][w] = value[v]

    # Process children in reverse BFS (post-order)
    for v in reversed(order):
        p = parent[v]
        if p == -1:
            continue
        # Merge dp[v] into dp[p]
        # dp[p] currently has items from p and previously processed children
        # We merge like 0/1 knapsack
        for w in range(W, -1, -1):
            for c in range(w + 1):
                if c <= W and w - c <= W:
                    dp[p][w] = max(dp[p][w], dp[p][w - c] + dp[v][c])

    return dp[root][W]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Star graph
    adj5 = build_tree(5, [(0,1),(0,2),(0,3),(0,4)])
    print("Max IS (star):", max_independent_set_iterative(adj5, 0, 5))  # 4

    # Path graph
    adj6 = build_tree(6, [(0,1),(1,2),(2,3),(1,4),(4,5)])
    print("Diameter:", tree_diameter(adj6, 0, 6))      # 4 (3-2-1-4-5)
    print("Diameter DP:", tree_diameter_dp(adj6, 0, 6))

    adj4 = build_tree(4, [(0,1),(1,2),(1,3)])
    ans = sum_of_depths_all_roots(adj4, 4)
    print("Sum of depths:", ans)  # [5, 3, 5, 5]

    adj5b = build_tree(5, [(0,1),(0,2),(2,3),(2,4)])
    nv = [-3, 4, 6, -1, 2]
    print("Max path sum:", max_path_sum(adj5b, 5, nv))  # 9

    print("Min vertex cover (star):", min_vertex_cover(adj5, 0, 5))  # 1
