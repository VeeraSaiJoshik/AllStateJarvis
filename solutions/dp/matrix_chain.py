"""
NAME: Matrix Chain Multiplication & Interval DP
TAGS: dp, interval-dp, matrix-chain, optimization
DESCRIPTION: Finds the optimal parenthesization of matrix multiplications to minimize
    scalar operations. This is the canonical interval DP problem — the same dp[i][j]
    over subproblems of increasing length pattern applies to polygon triangulation,
    optimal BST, and expression evaluation. Use whenever combining adjacent subproblems.
COMPLEXITY: Time O(n³), Space O(n²)
"""

import sys
sys.setrecursionlimit(10_000)

# ─────────────────────────────────────────────────────────────────────────────
# MATRIX CHAIN MULTIPLICATION — bottom-up O(n³)
# ─────────────────────────────────────────────────────────────────────────────
# dims[i] × dims[i+1] is the shape of matrix i (0-indexed, n matrices total).
# Cost of multiplying A(p×q) × B(q×r) = p*q*r scalar multiplications.

def matrix_chain(dims):
    """
    dims: list of n+1 integers where matrix i has shape dims[i] × dims[i+1].
    Returns (min_cost, split_table).

    Example:
        dims = [40, 20, 30, 10, 30]   # 4 matrices: 40×20, 20×30, 30×10, 10×30
        matrix_chain(dims) → (26000, split_table)
        # Optimal: ((A1 A2) A3) A4 is NOT optimal; check with backtrack below.
    """
    n = len(dims) - 1  # number of matrices
    # dp[i][j] = min cost to multiply matrices i..j (inclusive, 0-indexed)
    dp = [[0] * n for _ in range(n)]
    # split[i][j] = optimal split point k such that we split into [i..k][k+1..j]
    split = [[0] * n for _ in range(n)]

    # l = chain length (2 to n)
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = (dp[i][k] + dp[k + 1][j]
                        + dims[i] * dims[k + 1] * dims[j + 1])
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    return dp[0][n - 1], split


def matrix_chain_order(split, i, j, names=None):
    """
    Reconstructs the optimal parenthesization string.

    Example:
        dims = [40, 20, 30, 10, 30]
        cost, split = matrix_chain(dims)
        matrix_chain_order(split, 0, 3) → "((A0(A1A2))A3)"
    """
    if names is None:
        names = [f"A{k}" for k in range(len(split))]
    if i == j:
        return names[i]
    k = split[i][j]
    left  = matrix_chain_order(split, i, k, names)
    right = matrix_chain_order(split, k + 1, j, names)
    return f"({left}{right})"


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL INTERVAL DP TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
# Pattern:
#   for length in 2..n:
#     for i in 0..n-length:
#       j = i + length - 1
#       for k in i..j-1:
#         dp[i][j] = combine(dp[i][k], dp[k+1][j], cost(i,k,j))

def interval_dp_template(arr, cost_fn, combine_fn=min, base_val=0):
    """
    Generic interval DP.
    cost_fn(arr, i, k, j) → cost of merging subproblem [i..k] and [k+1..j].
    combine_fn: how to aggregate multiple split options (default: min).

    Returns dp table where dp[i][j] = optimal value for subarray arr[i..j].

    Example (stone merging):
        arr = [1, 2, 3, 4]
        # cost to merge segment = sum of elements in segment
        prefix = [0] + list(itertools.accumulate(arr))
        cost_fn = lambda arr, i, k, j: prefix[j+1] - prefix[i]
        dp = interval_dp_template(arr, cost_fn)
        # dp[0][3] = minimum cost to merge all 4 piles
    """
    n = len(arr)
    INF = float('inf')
    init_val = INF if combine_fn == min else -INF
    dp = [[init_val if i != j else base_val for j in range(n)] for i in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                val = cost_fn(arr, i, k, j)
                candidate = (dp[i][k] if dp[i][k] != init_val else base_val) + \
                            (dp[k+1][j] if dp[k+1][j] != init_val else base_val) + val
                if combine_fn == min:
                    dp[i][j] = min(dp[i][j], candidate)
                else:
                    dp[i][j] = max(dp[i][j], candidate)
    return dp


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMAL BINARY SEARCH TREE — classic interval DP
# Given sorted keys with search frequencies, minimize expected search cost.
# ─────────────────────────────────────────────────────────────────────────────

def optimal_bst(freq):
    """
    freq[i] = probability/frequency of searching for key i.
    Returns min expected cost (depth * freq sum).

    Example:
        freq = [0.25, 0.20, 0.05, 0.20, 0.30]
        optimal_bst(freq) → ~2.75
    """
    n = len(freq)
    # prefix sum for range sum queries
    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + freq[i]

    def range_sum(i, j):
        return prefix[j + 1] - prefix[i]

    dp = [[0.0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            rsum = range_sum(i, j)
            for k in range(i, j + 1):
                left  = dp[i][k - 1] if k > i else 0.0
                right = dp[k + 1][j] if k < j else 0.0
                cost  = left + right + rsum
                if cost < dp[i][j]:
                    dp[i][j] = cost
    return dp[0][n - 1]


# ─────────────────────────────────────────────────────────────────────────────
# POLYGON TRIANGULATION — minimum cost triangulation
# Cost of a triangle = product (or sum) of its vertices' values
# ─────────────────────────────────────────────────────────────────────────────

def min_triangulation_cost(values):
    """
    Given a convex polygon with n vertices, find min cost triangulation.
    Cost of triangle (i, k, j) = values[i] * values[k] * values[j].

    Example:
        min_triangulation_cost([1, 2, 3]) → 6         (only one triangle)
        min_triangulation_cost([1, 3, 1, 4, 1, 5]) → 13
    """
    n = len(values)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            dp[i][j] = float('inf')
            for k in range(i + 1, j):
                cost = (values[i] * values[k] * values[j]
                        + dp[i][k] + dp[k][j])
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n - 1]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dims = [40, 20, 30, 10, 30]
    cost, split = matrix_chain(dims)
    print("Matrix chain cost:", cost)                          # 26000
    print("Parenthesization:", matrix_chain_order(split, 0, 3))  # ((A0(A1A2))A3)

    freq = [0.25, 0.20, 0.05, 0.20, 0.30]
    print("Optimal BST cost:", optimal_bst(freq))             # ~2.75

    print("Triangle cost [1,2,3]:", min_triangulation_cost([1,2,3]))  # 6
    print("Triangle cost [1,3,1,4,1,5]:", min_triangulation_cost([1,3,1,4,1,5]))  # 13

    import itertools
    arr = [6, 2, 4, 3]
    prefix = [0] + list(itertools.accumulate(arr))
    # Stone merging: cost to merge segment is its sum
    cost_fn = lambda a, i, k, j: prefix[j+1] - prefix[i]
    dp = interval_dp_template(arr, cost_fn)
    print("Stone merge min cost:", dp[0][3])
