"""
NAME: LIS Variants (LDS, LNDS, LCIS, LBS, LWIS)
TAGS: dp, lis, subsequence, binary-search, variants
DESCRIPTION: Collection of LIS-family variants: Longest Decreasing Subsequence (LDS),
    Longest Non-Decreasing Subsequence (LNDS), Longest Common Increasing Subsequence
    (LCIS), Longest Bitonic Subsequence (LBS), and weighted LIS. Each variant appears
    frequently in competitive programming with only minor modifications to the core LIS DP.
COMPLEXITY: Time O(n log n) for LIS/LDS/LNDS; O(mn) for LCIS; O(n²) for LBS
"""

import bisect
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# 1. LONGEST DECREASING SUBSEQUENCE (LDS)
# ─────────────────────────────────────────────────────────────────────────────
# Reduce to LIS by negating elements or reversing.

def lds_length(nums):
    """
    Longest strictly decreasing subsequence length.  O(n log n).

    Trick: LDS(A) = LIS(-A)  or  LIS(A reversed).

    Example:
        lds_length([9, 4, 7, 2, 6, 3, 1, 5]) → 4  # [9,7,6,3] or [9,7,6,5] etc.
        lds_length([1, 2, 3, 4])              → 1
    """
    # negate so decreasing becomes increasing
    return _lis_strict([-x for x in nums])


def lds_sequence(nums):
    """
    Returns the actual LDS.

    Example:
        lds_sequence([9,4,7,2,6,3,1,5]) → [9,7,6,3] or similar
    """
    neg = [-x for x in nums]
    seq = _lis_sequence_impl(neg)
    return [-x for x in seq]


# ─────────────────────────────────────────────────────────────────────────────
# 2. LONGEST NON-DECREASING SUBSEQUENCE (LNDS) — allows equal elements
# ─────────────────────────────────────────────────────────────────────────────

def lnds_length(nums):
    """
    Longest non-decreasing subsequence.  O(n log n).
    Use bisect_right instead of bisect_left.

    Example:
        lnds_length([1, 3, 2, 2, 4]) → 4  # [1,2,2,4]
        lnds_length([5, 5, 5])       → 3
    """
    tails = []
    for x in nums:
        pos = bisect.bisect_right(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


def lnds_sequence(nums):
    """Returns the actual LNDS."""
    return _lis_sequence_impl(nums, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LONGEST COMMON INCREASING SUBSEQUENCE (LCIS)
# ─────────────────────────────────────────────────────────────────────────────
# LCIS = longest subsequence that is common to A and B AND strictly increasing.
# O(mn) DP.

def lcis_length(a, b):
    """
    Returns the LCIS length of arrays a and b.

    dp[j] = length of LCIS of a[:i] and b[:j] ending at b[j].

    Example:
        lcis_length([3,4,9,1], [5,3,8,9,10,2,1]) → 2  # [3,9] or [1,...] wait:
        # Common increasing: [3,9], [3,10]? 10 not in a. [9,?] only [9].
        # a=[3,4,9,1], b=[5,3,8,9,10,2,1]: common: 3,9,1.
        # Increasing common subseq: [3,9] len=2.  → 2
        lcis_length([1,3,4,5,6,2,7,8], [1,2,3,5,7,4,6,8]) → 5  # [1,3,5,6,8]? verify
    """
    m, n = len(a), len(b)
    dp = [0] * n   # dp[j] = LCIS length ending at b[j]

    for i in range(m):
        cur = 0  # current best LCIS length ending before b[j]
        for j in range(n):
            if a[i] == b[j]:
                dp[j] = max(dp[j], cur + 1)
            elif b[j] < a[i]:
                cur = max(cur, dp[j])
    return max(dp) if dp else 0


def lcis_sequence(a, b):
    """
    Returns the actual LCIS (one valid answer).

    Example:
        lcis_sequence([3,4,9,1], [5,3,8,9,10,2,1]) → [3,9]
    """
    m, n = len(a), len(b)
    dp   = [0] * n
    prev = [-1]  * n   # previous index in b for reconstruction

    for i in range(m):
        cur     = 0
        cur_idx = -1
        for j in range(n):
            if a[i] == b[j] and cur + 1 > dp[j]:
                dp[j]   = cur + 1
                prev[j] = cur_idx
            if b[j] < a[i] and dp[j] > cur:
                cur     = dp[j]
                cur_idx = j

    # find end of LCIS
    end_j = max(range(n), key=lambda j: dp[j]) if n > 0 else -1
    result = []
    j = end_j
    while j != -1:
        result.append(b[j])
        j = prev[j]
    result.reverse()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. LONGEST BITONIC SUBSEQUENCE (LBS)
# ─────────────────────────────────────────────────────────────────────────────
# Bitonic = first strictly increasing then strictly decreasing.
# LBS[i] = LIS ending at i + LDS starting at i - 1

def lbs_length(nums):
    """
    Returns the length of the longest bitonic subsequence.

    Example:
        lbs_length([1,11,2,10,4,5,2,1]) → 6  # [1,2,10,5,2,1] or [1,2,4,5,2,1]
        lbs_length([1,2,5,3,2])         → 5  # entire array
    """
    n = len(nums)
    # LIS ending at each index (from left)
    lis = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                lis[i] = max(lis[i], lis[j] + 1)
    # LDS starting at each index = LIS from right
    lds = [1] * n
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if nums[j] < nums[i]:
                lds[i] = max(lds[i], lds[j] + 1)
    # Bitonic at i requires at least 2 elements on each side
    best = 0
    for i in range(n):
        if lis[i] > 1 and lds[i] > 1:  # true bitonic (goes up then down)
            best = max(best, lis[i] + lds[i] - 1)
    # If no true bitonic, return the pure LIS or LDS
    if best == 0:
        best = max(max(lis), max(lds))
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 5. WEIGHTED LIS — maximize sum of values (not just count)
# ─────────────────────────────────────────────────────────────────────────────

def weighted_lis(nums, weights):
    """
    Find increasing subsequence maximizing sum of weights.
    O(n log n) using a sorted structure.

    Simpler O(n²) version for clarity.

    Example:
        nums    = [1, 3, 2, 5, 4]
        weights = [2, 5, 3, 7, 6]
        weighted_lis(nums, weights) → 14  # [1,3,5] weights [2,5,7] = 14
    """
    n = len(nums)
    dp = list(weights)   # dp[i] = max weight sum of IS ending at i
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + weights[i])
    return max(dp)


# ─────────────────────────────────────────────────────────────────────────────
# 6. LONGEST INCREASING SUBSEQUENCE IN CIRCULAR ARRAY
# ─────────────────────────────────────────────────────────────────────────────

def lis_circular(nums):
    """
    LIS in a circular arrangement: concatenate the array with itself
    and find LIS of length ≤ n (to avoid using more than n elements).

    Example:
        lis_circular([3, 1, 4, 2]) → 3  # [1,4,...] or [1,2,...] cycling through
    """
    n = len(nums)
    doubled = nums + nums
    # O(n²) approach tracking which original indices are used
    m = len(doubled)
    dp = [1] * m
    for i in range(1, m):
        for j in range(i):
            if doubled[j] < doubled[i] and (i - j) < n:
                dp[i] = max(dp[i], dp[j] + 1)
    return min(max(dp), n)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MINIMUM NUMBER OF CHAINS TO PARTITION INTO NON-INCREASING SUBSEQUENCES
# (Dilworth's theorem: min chains = max antichain = LIS length)
# ─────────────────────────────────────────────────────────────────────────────

def min_chains_partition(nums):
    """
    Minimum number of non-increasing subsequences needed to partition nums.
    By Dilworth's theorem = LIS length of nums.

    Example:
        min_chains_partition([3,1,4,1,5,9,2,6]) → LIS length
    """
    return _lis_strict(nums)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lis_strict(nums):
    """LIS length (strictly increasing) O(n log n)."""
    tails = []
    for x in nums:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


def _lis_sequence_impl(nums, strict=True):
    """Returns actual LIS sequence O(n log n)."""
    n = len(nums)
    if n == 0:
        return []
    tails     = []
    tail_idx  = []
    parent    = [-1] * n

    for i, x in enumerate(nums):
        if strict:
            pos = bisect.bisect_left(tails, x)
        else:
            pos = bisect.bisect_right(tails, x)
        if pos == len(tails):
            tails.append(x)
            tail_idx.append(i)
        else:
            tails[pos] = x
            tail_idx[pos] = i
        parent[i] = tail_idx[pos - 1] if pos > 0 else -1

    result = []
    idx = tail_idx[-1]
    while idx != -1:
        result.append(nums[idx])
        idx = parent[idx]
    result.reverse()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    A = [9, 4, 7, 2, 6, 3, 1, 5]
    print("LDS length:", lds_length(A))          # 4
    print("LDS seq:", lds_sequence(A))           # e.g. [9,7,6,3]

    B = [1, 3, 2, 2, 4]
    print("LNDS:", lnds_length(B))               # 4

    a = [3, 4, 9, 1]
    b = [5, 3, 8, 9, 10, 2, 1]
    print("LCIS len:", lcis_length(a, b))        # 2
    print("LCIS seq:", lcis_sequence(a, b))      # [3, 9]

    C = [1, 11, 2, 10, 4, 5, 2, 1]
    print("LBS:", lbs_length(C))                 # 6

    nums    = [1, 3, 2, 5, 4]
    weights = [2, 5, 3, 7, 6]
    print("Weighted LIS:", weighted_lis(nums, weights))  # 14
