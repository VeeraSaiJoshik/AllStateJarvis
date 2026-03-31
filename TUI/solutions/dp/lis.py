"""
NAME: Longest Increasing Subsequence (LIS) — O(n log n)
TAGS: dp, binary-search, patience-sorting, greedy
DESCRIPTION: Finds the length of the longest strictly increasing subsequence using
    patience sorting with binary search. This is the gold-standard O(n log n) approach
    used in competitions. Also includes O(n²) DP when the actual subsequence is needed.
COMPLEXITY: Time O(n log n), Space O(n)
"""

import bisect

# ─────────────────────────────────────────────────────────────────────────────
# LIS LENGTH — O(n log n) patience sorting
# ─────────────────────────────────────────────────────────────────────────────
# 'tails[i]' = smallest tail element of all increasing subsequences of length i+1.
# tails is always sorted; we binary-search to find where current element fits.

def lis_length(nums):
    """
    Returns the length of the LIS (strictly increasing).

    Example:
        lis_length([10, 9, 2, 5, 3, 7, 101, 18]) → 4  # [2,3,7,18] or [2,5,7,101]
        lis_length([0, 1, 0, 3, 2, 3])            → 4  # [0,1,2,3] or [0,1,3]
        lis_length([7, 7, 7, 7])                  → 1
    """
    tails = []
    for x in nums:
        pos = bisect.bisect_left(tails, x)   # strict: replace equal
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


def lis_length_nondecreasing(nums):
    """
    Returns the length of the longest non-decreasing subsequence.
    Change: use bisect_right so equal elements extend the subsequence.

    Example:
        lis_length_nondecreasing([7, 7, 7, 7]) → 4
    """
    tails = []
    for x in nums:
        pos = bisect.bisect_right(tails, x)  # non-strict: allow equals
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


# ─────────────────────────────────────────────────────────────────────────────
# LIS WITH RECONSTRUCTION — O(n log n) time, returns actual subsequence
# ─────────────────────────────────────────────────────────────────────────────

def lis_sequence(nums):
    """
    Returns the actual LIS (one valid answer, lexicographically smallest tail).

    Uses auxiliary arrays:
      tails[i] = smallest tail for IS of length i+1
      tail_idx[i] = index in nums of tails[i]
      parent[i]   = index in nums of the element before nums[i] in the IS

    Example:
        lis_sequence([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]) → [1, 2, 5, 9] or similar
    """
    n = len(nums)
    if n == 0:
        return []

    # tails stores actual VALUES (for binary search)
    tails = []
    tail_idx = []     # which index in nums does tails[k] come from
    parent = [-1] * n # parent[i] = predecessor index of nums[i] in IS

    for i, x in enumerate(nums):
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
            tail_idx.append(i)
        else:
            tails[pos] = x
            tail_idx[pos] = i
        # predecessor is the element at tails[pos-1]
        parent[i] = tail_idx[pos - 1] if pos > 0 else -1

    # reconstruct: start from tail_idx[-1] (last element of LIS)
    result = []
    idx = tail_idx[-1]
    while idx != -1:
        result.append(nums[idx])
        idx = parent[idx]
    result.reverse()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LIS — classic O(n²) DP (easier to modify for custom comparators)
# ─────────────────────────────────────────────────────────────────────────────

def lis_dp(nums):
    """
    O(n²) DP.  dp[i] = length of LIS ending at index i.
    Returns (length, dp_array).  dp_array useful for extensions.

    Example:
        lis_dp([10, 9, 2, 5, 3, 7, 101, 18]) → (4, [1,1,1,2,2,3,4,4])
    """
    n = len(nums)
    if n == 0:
        return 0, []
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp), dp


# ─────────────────────────────────────────────────────────────────────────────
# NUMBER OF LIS — count distinct LIS of maximum length  O(n²)
# ─────────────────────────────────────────────────────────────────────────────

def count_lis(nums):
    """
    Returns the number of LIS of maximum length.

    Example:
        count_lis([1, 3, 5, 4, 7]) → 2  # [1,3,5,7] and [1,3,4,7]
        count_lis([2, 2, 2, 2])    → 1  # only [2]
    """
    n = len(nums)
    if n == 0:
        return 0
    length = [1] * n  # LIS length ending at i
    count  = [1] * n  # number of LIS ending at i
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if length[j] + 1 > length[i]:
                    length[i] = length[j] + 1
                    count[i]  = count[j]
                elif length[j] + 1 == length[i]:
                    count[i] += count[j]
    max_len = max(length)
    return sum(c for l, c in zip(length, count) if l == max_len)


# ─────────────────────────────────────────────────────────────────────────────
# PATIENCE SORTING — full card pile simulation (educational)
# ─────────────────────────────────────────────────────────────────────────────

def patience_sort_piles(nums):
    """
    Returns list of piles (each pile is a list of cards).
    Number of piles = LIS length.  Useful for visualizing the algorithm.

    Example:
        patience_sort_piles([3, 1, 4, 1, 5, 9, 2, 6]) →
        [[1, 1], [2, 4], [5, 6], [9]] or similar  (tops are non-decreasing)
    """
    piles = []   # each pile stores all cards (top is piles[k][-1])
    tops  = []   # tops[k] = top card of pile k (sorted)
    for x in nums:
        pos = bisect.bisect_left(tops, x)
        if pos == len(piles):
            piles.append([x])
            tops.append(x)
        else:
            piles[pos].append(x)
            tops[pos] = x
    return piles


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    A = [10, 9, 2, 5, 3, 7, 101, 18]
    print("LIS length:", lis_length(A))                 # 4
    print("LIS sequence:", lis_sequence(A))             # e.g. [2, 3, 7, 18]
    print("LIS dp:", lis_dp(A))                         # (4, [...])
    print("Count LIS:", count_lis([1, 3, 5, 4, 7]))    # 2
    print("Non-decr:", lis_length_nondecreasing([7,7,7,7]))  # 4
    print("Piles:", patience_sort_piles([3,1,4,1,5,9,2,6]))
