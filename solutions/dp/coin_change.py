"""
NAME: Coin Change DP (Min Coins & Number of Ways)
TAGS: dp, coins, unbounded-knapsack, combinatorics
DESCRIPTION: Two classic coin problems: (1) minimum number of coins to make a target
    amount, and (2) count the number of distinct ways to make the amount. Both are
    unbounded knapsack variants — use when competitions involve denominations/amounts.
COMPLEXITY: Time O(n * amount), Space O(amount)
"""

import math

# ─────────────────────────────────────────────────────────────────────────────
# COIN CHANGE 1 — minimum number of coins to reach exactly `amount`
# ─────────────────────────────────────────────────────────────────────────────

def coin_change_min(coins, amount):
    """
    Returns the fewest coins needed to make `amount`, or -1 if impossible.

    dp[a] = min coins to make amount a.
    Transition: dp[a] = min(dp[a - c] + 1) for each coin c <= a.

    Example:
        coin_change_min([1, 5, 6, 9], 11) → 2  (5+6)
        coin_change_min([2], 3)            → -1
        coin_change_min([1, 2, 5], 11)    → 3  (5+5+1)
    """
    INF = float('inf')
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != INF else -1


def coin_change_min_with_coins(coins, amount):
    """
    Returns (min_count, list_of_coins_used), or (-1, []) if impossible.

    Example:
        coin_change_min_with_coins([1,5,6,9], 11) → (2, [5, 6])
    """
    INF = float('inf')
    dp = [INF] * (amount + 1)
    dp[0] = 0
    last_coin = [-1] * (amount + 1)
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
                last_coin[a] = c
    if dp[amount] == INF:
        return -1, []
    # reconstruct
    used = []
    cur = amount
    while cur > 0:
        used.append(last_coin[cur])
        cur -= last_coin[cur]
    return dp[amount], sorted(used)


# ─────────────────────────────────────────────────────────────────────────────
# COIN CHANGE 2 — number of ways (combinations, order doesn't matter)
# ─────────────────────────────────────────────────────────────────────────────
# KEY: outer loop over coins, inner over amounts → counts combinations only
# (each combination counted once regardless of order).

def coin_change_ways(coins, amount):
    """
    Returns the number of distinct combinations of coins that sum to `amount`.
    Each coin denomination can be used unlimited times.
    Order does NOT matter: {1,2} == {2,1}.

    Example:
        coin_change_ways([1,2,5], 5) → 4
        # Ways: [5], [2,2,1], [2,1,1,1], [1,1,1,1,1]
        coin_change_ways([2], 3)     → 0
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:           # outer = coins → ensures each subset counted once
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]


def coin_change_permutations(coins, amount):
    """
    Returns the number of ordered sequences (permutations) of coins summing to `amount`.
    Order DOES matter: {1,2} != {2,1}.

    Example:
        coin_change_permutations([1,2,3], 4) → 7
        # [1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 1+3, 3+1, 2+2]
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    for a in range(1, amount + 1):    # outer = amounts → counts all orderings
        for c in coins:
            if c <= a:
                dp[a] += dp[a - c]
    return dp[amount]


# ─────────────────────────────────────────────────────────────────────────────
# COIN CHANGE — exact count with limited supply (0/1 coins)
# ─────────────────────────────────────────────────────────────────────────────

def coin_change_01(coins, amount):
    """
    Each coin can be used AT MOST ONCE.
    Returns min coins, or -1 if impossible.

    Example:
        coin_change_01([1,1,1,2,5], 7) → 2  # 5+2
        coin_change_01([1,2,5], 7)     → 2  # 5+2 (but each used at most once → still 2)
    """
    INF = float('inf')
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for c in coins:
        # right-to-left (0/1 knapsack style)
        for a in range(amount, c - 1, -1):
            if dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != INF else -1


def coin_change_ways_01(coins, amount):
    """
    Count combinations where each coin is used at most once.

    Example:
        coin_change_ways_01([1,2,3,4,5], 5) → 3
        # {5}, {1,4}, {2,3}
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for a in range(amount, c - 1, -1):
            dp[a] += dp[a - c]
    return dp[amount]


# ─────────────────────────────────────────────────────────────────────────────
# MINIMUM COIN CHANGE — BFS approach (finds shortest path in coin graph)
# Useful when coins are very large but amount is small, or for competitions
# where BFS is more natural.
# ─────────────────────────────────────────────────────────────────────────────

from collections import deque

def coin_change_bfs(coins, amount):
    """
    BFS-based coin change: finds minimum coins via BFS level = coin count.
    Equivalent result to coin_change_min but uses BFS.

    Example:
        coin_change_bfs([1,5,6,9], 11) → 2
    """
    if amount == 0:
        return 0
    visited = [False] * (amount + 1)
    visited[0] = True
    q = deque([0])
    steps = 0
    while q:
        steps += 1
        for _ in range(len(q)):
            cur = q.popleft()
            for c in coins:
                nxt = cur + c
                if nxt == amount:
                    return steps
                if nxt < amount and not visited[nxt]:
                    visited[nxt] = True
                    q.append(nxt)
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Min coins [1,5,6,9] → 11:", coin_change_min([1,5,6,9], 11))         # 2
    print("Min coins [1,2,5]  → 11:", coin_change_min([1,2,5], 11))            # 3
    print("Min coins [2]      → 3:",  coin_change_min([2], 3))                 # -1
    cnt, used = coin_change_min_with_coins([1,5,6,9], 11)
    print(f"  used: {used}")                                                    # [5,6]

    print("\nWays [1,2,5] → 5:",  coin_change_ways([1,2,5], 5))               # 4
    print("Ways [2]     → 3:",    coin_change_ways([2], 3))                    # 0
    print("Perms [1,2,3] → 4:", coin_change_permutations([1,2,3], 4))         # 7

    print("\n0/1 ways [1,2,3,4,5] → 5:", coin_change_ways_01([1,2,3,4,5], 5)) # 3

    print("\nBFS [1,5,6,9] → 11:", coin_change_bfs([1,5,6,9], 11))            # 2
