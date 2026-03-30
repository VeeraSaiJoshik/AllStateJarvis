"""
NAME: Bitmask DP (TSP, Assignment, Subset Enumeration)
TAGS: dp, bitmask, tsp, assignment, combinatorics
DESCRIPTION: Uses bitmasks to compactly represent subsets of elements as DP states.
    Essential for problems with n ≤ 20 where you need to track which elements have
    been used. Covers TSP (shortest Hamiltonian path/cycle), assignment problems,
    and counting/optimizing over subsets.
COMPLEXITY: Time O(2^n * n²) for TSP/assignment, Space O(2^n * n)
"""

import sys
from functools import lru_cache

INF = float('inf')

# ─────────────────────────────────────────────────────────────────────────────
# BITMASK BASICS — helpers for bit manipulation
# ─────────────────────────────────────────────────────────────────────────────

def bit(mask, i):        return (mask >> i) & 1          # get bit i
def set_bit(mask, i):    return mask | (1 << i)           # set bit i
def clear_bit(mask, i):  return mask & ~(1 << i)          # clear bit i
def toggle_bit(mask, i): return mask ^ (1 << i)           # toggle bit i
def popcount(mask):      return bin(mask).count('1')      # number of set bits
def lowest_bit(mask):    return mask & (-mask)            # isolate lowest set bit
def lowest_bit_idx(mask):return (mask & -mask).bit_length() - 1  # index of lowest set bit

def submasks(mask):
    """Enumerate all non-empty submasks of mask in O(2^popcount) time."""
    sub = mask
    while sub:
        yield sub
        sub = (sub - 1) & mask

def all_subsets_of_size_k(n, k):
    """Generate all bitmasks of n bits with exactly k bits set (Gosper's hack)."""
    if k == 0:
        yield 0
        return
    mask = (1 << k) - 1  # smallest k-bit number
    limit = 1 << n
    while mask < limit:
        yield mask
        # Gosper's hack: next permutation of bits
        c = mask & -mask
        r = mask + c
        mask = (((r ^ mask) >> 2) // c) | r


# ─────────────────────────────────────────────────────────────────────────────
# TRAVELLING SALESMAN PROBLEM (TSP) — Held-Karp algorithm
# Find the shortest Hamiltonian cycle through n cities.
# ─────────────────────────────────────────────────────────────────────────────

def tsp(dist):
    """
    dist[i][j] = cost of traveling from city i to city j.
    Returns (min_cost, path) of the shortest Hamiltonian cycle starting at 0.

    Time: O(2^n * n²)   Space: O(2^n * n)   Limit: n ≤ 20 (practical ≤ 15-18)

    Example:
        dist = [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
        tsp(dist) → (80, [0, 1, 3, 2, 0])
    """
    n = len(dist)
    if n == 1:
        return 0, [0]
    FULL = (1 << n) - 1

    # dp[mask][i] = min cost to visit all cities in mask, ending at city i
    # starting from city 0
    dp   = [[INF] * n for _ in range(1 << n)]
    prev = [[-1]  * n for _ in range(1 << n)]

    dp[1][0] = 0   # start at city 0, visited = {0}

    for mask in range(1, 1 << n):
        for u in range(n):
            if not bit(mask, u) or dp[mask][u] == INF:
                continue
            for v in range(n):
                if bit(mask, v):
                    continue  # already visited
                new_mask = set_bit(mask, v)
                new_cost = dp[mask][u] + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    prev[new_mask][v] = u

    # Return to start
    best_cost = INF
    last_city = -1
    for u in range(1, n):
        cost = dp[FULL][u] + dist[u][0]
        if cost < best_cost:
            best_cost = cost
            last_city = u

    # Reconstruct path
    path = []
    mask = FULL
    cur  = last_city
    while cur != -1:
        path.append(cur)
        nxt = prev[mask][cur]
        mask = clear_bit(mask, cur)
        cur  = nxt
    path.reverse()
    path.append(0)   # return to start

    return best_cost, path


def tsp_memo(dist):
    """
    Recursive + memoization version of TSP (same complexity, often cleaner).

    Example:
        dist = [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
        tsp_memo(dist) → 80
    """
    n = len(dist)
    FULL = (1 << n) - 1

    @lru_cache(maxsize=None)
    def dp(mask, u):
        if mask == FULL:
            return dist[u][0]   # return to start
        best = INF
        for v in range(n):
            if not bit(mask, v):
                best = min(best, dist[u][v] + dp(set_bit(mask, v), v))
        return best

    return dp(1, 0)   # start at city 0, visited = {0}


# ─────────────────────────────────────────────────────────────────────────────
# ASSIGNMENT PROBLEM — minimum cost perfect matching (workers ↔ tasks)
# ─────────────────────────────────────────────────────────────────────────────

def assignment_problem(cost):
    """
    n workers × n tasks.  cost[i][j] = cost of assigning worker i to task j.
    Returns (min_total_cost, assignment_list) where assignment[i] = task for worker i.

    Bitmask DP: dp[mask] = min cost when first popcount(mask) workers are assigned
    to the tasks indicated by mask.

    Example:
        cost = [[9,2,7,8],[6,4,3,7],[5,8,1,8],[7,6,9,4]]
        assignment_problem(cost) → (13, [1, 0, 2, 3])
        # Worker 0→task1(cost2), worker1→task0(cost6), worker2→task2(cost1), worker3→task3(cost4) = 13
    """
    n = len(cost)
    dp   = [INF] * (1 << n)
    from_task = [-1] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue
        worker = popcount(mask)   # next worker to assign
        if worker == n:
            continue
        for task in range(n):
            if bit(mask, task):
                continue          # task already assigned
            new_mask = set_bit(mask, task)
            new_cost = dp[mask] + cost[worker][task]
            if new_cost < dp[new_mask]:
                dp[new_mask] = new_cost
                from_task[new_mask] = task

    # Reconstruct
    assignment = [-1] * n
    mask = (1 << n) - 1
    for worker in range(n - 1, -1, -1):
        task = from_task[mask]
        assignment[worker] = task
        mask = clear_bit(mask, task)

    return dp[(1 << n) - 1], assignment


# ─────────────────────────────────────────────────────────────────────────────
# COUNTING SUBSETS WITH PROPERTY — Sum over subsets (SOS DP)
# ─────────────────────────────────────────────────────────────────────────────

def sum_over_subsets(a, n):
    """
    Computes f[mask] = sum of a[sub] for all submasks sub of mask.
    Uses the SOS (Sum over Subsets) DP in O(n * 2^n).

    Example:
        a = [1, 2, 3, 4]   # indexed by 2-bit mask (n=2)
        sum_over_subsets(a, 2) → [1, 3, 4, 10]
        # f[0]=a[0]=1; f[1]=a[0]+a[1]=3; f[2]=a[0]+a[2]=4; f[3]=all=10
    """
    f = list(a)  # copy
    for i in range(n):
        for mask in range(1 << n):
            if bit(mask, i):
                f[mask] += f[mask ^ (1 << i)]
    return f


def max_over_subsets(a, n):
    """
    Computes f[mask] = max of a[sub] for all submasks sub of mask.
    """
    f = list(a)
    for i in range(n):
        for mask in range(1 << n):
            if bit(mask, i):
                f[mask] = max(f[mask], f[mask ^ (1 << i)])
    return f


# ─────────────────────────────────────────────────────────────────────────────
# BITMASK DP — cover all elements using given subsets
# (Set Cover / Exact Cover)
# ─────────────────────────────────────────────────────────────────────────────

def min_set_cover(n, subsets):
    """
    n elements (bits 0..n-1), subsets = list of bitmasks.
    Returns minimum number of subsets needed to cover all n elements, or -1.

    Example:
        n = 3  # elements {0,1,2}
        subsets = [0b011, 0b101, 0b110, 0b111]  # {0,1},{0,2},{1,2},{0,1,2}
        min_set_cover(3, subsets) → 1  # use {0,1,2}
    """
    FULL = (1 << n) - 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    for mask in range(1, 1 << n):
        for s in subsets:
            prev = mask & ~s   # mask without the bits covered by s
            if dp[prev] + 1 < dp[mask]:
                dp[mask] = dp[prev] + 1
    return dp[FULL] if dp[FULL] != INF else -1


def count_set_cover_ways(n, subsets):
    """
    Count the number of distinct subset selections that cover all n elements
    (each subset used at most once).

    Example:
        n = 2  # elements {0,1}
        subsets = [0b01, 0b10, 0b11]  # {0},{1},{0,1}
        count_set_cover_ways(2, subsets) → 3
        # {0,1}, {0}+{1}+{0,1}? actually: {0}+{1}=1 way, {0,1}=1 way, {0}+{1,0}=1 way → 3
    """
    FULL = (1 << n) - 1
    dp = [0] * (1 << n)
    dp[0] = 1
    for i, s in enumerate(subsets):
        # 0/1 choice: include subset i or not
        for mask in range(FULL, -1, -1):
            new_mask = mask | s
            dp[new_mask] += dp[mask]
    return dp[FULL]


# ─────────────────────────────────────────────────────────────────────────────
# BITMASK DP — Hamiltonian path (no return to start)
# ─────────────────────────────────────────────────────────────────────────────

def shortest_hamiltonian_path(dist, start=0):
    """
    Finds the shortest Hamiltonian PATH starting at `start` (no return).
    Returns (min_cost, end_city).

    Example:
        dist = [[0,1,2],[1,0,3],[2,3,0]]
        shortest_hamiltonian_path(dist, 0) → (3, 2)  # path 0→1→2 cost=1+3=4? or 0→2→1=2+3=5; 0→1→2=4 Hmm
        # Actually 0→2→1 = dist[0][2]+dist[2][1] = 2+3=5; 0→1→2 = 1+3=4 → (4, 2)
    """
    n = len(dist)
    FULL = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1 << start][start] = 0

    for mask in range(1 << n):
        for u in range(n):
            if not bit(mask, u) or dp[mask][u] == INF:
                continue
            for v in range(n):
                if bit(mask, v):
                    continue
                new_mask = set_bit(mask, v)
                new_cost = dp[mask][u] + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost

    best = INF
    end  = -1
    for u in range(n):
        if u != start and dp[FULL][u] < best:
            best = dp[FULL][u]
            end  = u
    return best, end


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TSP
    dist = [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
    cost, path = tsp(dist)
    print("TSP cost:", cost, "path:", path)      # 80, [0,1,3,2,0]
    print("TSP memo:", tsp_memo(dist))           # 80

    # Assignment
    c = [[9,2,7,8],[6,4,3,7],[5,8,1,8],[7,6,9,4]]
    total, assign = assignment_problem(c)
    print("Assignment:", total, assign)          # 13

    # SOS DP
    a = [1, 2, 3, 4]
    print("SOS:", sum_over_subsets(a, 2))        # [1,3,4,10]

    # Set cover
    print("Min set cover:", min_set_cover(3, [0b011,0b101,0b110,0b111]))  # 1

    # Gosper's hack
    print("Subsets of size 2 from 4:", list(all_subsets_of_size_k(4, 2)))
