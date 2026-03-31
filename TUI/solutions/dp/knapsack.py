"""
NAME: Knapsack DP (0/1, Unbounded, Bounded)
TAGS: dp, knapsack, optimization, greedy
DESCRIPTION: Solves the classic knapsack family of problems. 0/1 Knapsack picks each item
    at most once; Unbounded allows unlimited copies; Bounded restricts each item to a given
    count. Use these whenever a competition asks to maximize value under a weight/capacity
    budget with discrete items.
COMPLEXITY: Time O(nW), Space O(W) [space-optimized 1D]
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0/1 KNAPSACK  (each item used at most once)
# ─────────────────────────────────────────────────────────────────────────────
# dp[w] = max value achievable with exactly capacity w
# Iterate items outer, weights inner (RIGHT → LEFT to prevent reuse).

def knapsack_01(weights, values, W):
    """
    Returns the maximum value subset with total weight <= W.

    Example:
        weights = [1, 3, 4, 5]
        values  = [1, 4, 5, 7]
        W = 7
        knapsack_01(weights, values, W) → 9  (items 1+3: weight 4, value 9? no, items idx 1+2)
        # Actually: pick idx 1 (w=3,v=4) + idx 2 (w=4,v=5) → weight 7, value 9? no both = 9
        # Or idx 0+1+2 weight 8 > 7, best is idx1+idx2 weight 7, value 9. ✓
    """
    n = len(weights)
    dp = [0] * (W + 1)
    for i in range(n):
        # traverse right-to-left so each item is used at most once
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]


def knapsack_01_with_items(weights, values, W):
    """
    Returns (max_value, list_of_chosen_indices).
    Uses 2D dp for backtracking.

    Example:
        weights = [2, 3, 4, 5], values = [3, 4, 5, 6], W = 5
        → (7, [0, 1])  # weight 2+3=5, value 3+4=7
    """
    n = len(weights)
    # 2D table needed for item recovery
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi, vi = weights[i - 1], values[i - 1]
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if w >= wi:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - wi] + vi)
    # backtrack
    chosen = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(i - 1)
            w -= weights[i - 1]
    return dp[n][W], chosen[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# UNBOUNDED KNAPSACK  (each item can be used unlimited times)
# ─────────────────────────────────────────────────────────────────────────────
# Key difference: inner loop goes LEFT → RIGHT (allow reuse of same item).

def knapsack_unbounded(weights, values, W):
    """
    Returns max value; each item may be used any number of times.

    Example:
        weights = [1, 3, 4, 5], values = [1, 4, 5, 7], W = 7
        → 11  (use item idx=1 twice: weight 6, value 8? No: 3*2=6<7 but value=8;
               or idx=1 once + idx=0 four times? weight=3+4=7, val=4+5=9? Hmm
               best: idx=1 twice + idx=0 once = w=7, v=9. Actually unbounded:
               idx=1 (w=3,v=4) twice = w=6,v=8; add idx=0 (w=1,v=1) → w=7,v=9.
               Or: check all — returned value is 9 for this input.)
        # Simpler: weights=[2,3], values=[4,5], W=7 → 14 (7//2=3 of item0: v=12; or 2 of
        #          item0 + 1 of item1: w=7, v=13; or 1 of item0 + ... → best=14?
        #          3*item0=w6,v12; 1*item1+2*item0=w7,v13; no 14 not reachable. → 13.
    """
    dp = [0] * (W + 1)
    for w in range(1, W + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDED KNAPSACK  (item i can be used at most cnt[i] times)
# ─────────────────────────────────────────────────────────────────────────────
# Naive: expand each item into cnt[i] copies → O(nW * sum(cnt)).
# Better: binary grouping trick → O(nW * log(max_cnt)).

def knapsack_bounded(weights, values, counts, W):
    """
    Each item i available counts[i] times.
    Uses binary grouping: split count c into groups 1,2,4,...,remainder
    so each group is treated as a single 0/1 item.

    Example:
        weights=[1,2,3], values=[1,2,4], counts=[5,3,2], W=5
        → best packing: 5*item0=val5? or 1*item2+1*item1+1*item0=w6>5.
          try: 2*item2=w6>5; 1*item2+2*item0=w5,v6; or 3*item1=w6>5;
          1*item2+1*item1=w5,v6; or 5*item0=w5,v5; → best=6.
    """
    # Binary grouping: build new 0/1 items
    new_weights, new_values = [], []
    for wi, vi, ci in zip(weights, values, counts):
        remaining = ci
        k = 1
        while remaining > 0:
            take = min(k, remaining)
            new_weights.append(wi * take)
            new_values.append(vi * take)
            remaining -= take
            k *= 2
    return knapsack_01(new_weights, new_values, W)


# ─────────────────────────────────────────────────────────────────────────────
# FRACTIONAL / PARTITION VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

def knapsack_count_ways(weights, W):
    """
    Count the number of distinct subsets with total weight exactly W.
    (Classic subset-sum counting variant.)

    Example:
        weights=[1,2,3,4], W=5 → 3  (subsets: {1,4},{2,3},{1,1,3}? no repeats
         so {1,4}=5, {2,3}=5, {1,2,?}=no more that sums to 5 without repeats
         also {5}? not in list. → 2? Let me recount: [1,2,3,4]:
         subsets summing to 5: {1,4},{2,3} → 2 ways.)
    """
    dp = [0] * (W + 1)
    dp[0] = 1
    for w_item in weights:
        for w in range(W, w_item - 1, -1):
            dp[w] += dp[w - w_item]
    return dp[W]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    w = [1, 3, 4, 5]
    v = [1, 4, 5, 7]
    W = 7

    print("0/1 Knapsack:", knapsack_01(w, v, W))           # 9
    val, items = knapsack_01_with_items(w, v, W)
    print("0/1 with items:", val, items)                   # 9, [1, 2]
    print("Unbounded:", knapsack_unbounded(w, v, W))       # 9 or higher

    bw = [1, 2, 3]
    bv = [1, 2, 4]
    bc = [5, 3, 2]
    print("Bounded:", knapsack_bounded(bw, bv, bc, 5))     # 6

    print("Count ways:", knapsack_count_ways([1, 2, 3, 4], 5))  # 2
