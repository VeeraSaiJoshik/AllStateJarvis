"""
NAME: Suffix Array with LCP Array
TAGS: string, suffix array, LCP, O(n log n), substring problems
DESCRIPTION: Builds the suffix array (sorted order of all suffixes) in O(n log n) using
             prefix-doubling, then computes the LCP array via Kasai's algorithm in O(n).
             Essential for problems involving distinct substrings, longest repeated substring,
             longest common substring of multiple strings, and string compression.
COMPLEXITY: Time O(n log n) build + O(n) LCP, Space O(n)
"""

# --------------------------------------------------------------------------- #
#  Suffix Array — O(n log n) prefix-doubling (DC3/SA-IS is faster but complex)
# --------------------------------------------------------------------------- #

def build_suffix_array(s: str) -> list[int]:
    """
    Returns the suffix array SA where SA[i] is the starting index of the
    i-th lexicographically smallest suffix of s.

    Append a sentinel character smaller than all others (e.g. '$') before calling
    if needed, or the implementation handles it via Python string comparison.

    Example:
        build_suffix_array("banana")
        -> [5, 3, 1, 0, 4, 2]
           suffixes in order: "a","ana","anana","banana","na","nana"
    """
    n = len(s)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Initial rank based on character value
    sa = sorted(range(n), key=lambda i: s[i])
    rank = [0] * n
    rank[sa[0]] = 0
    for i in range(1, n):
        rank[sa[i]] = rank[sa[i - 1]] + (s[sa[i]] != s[sa[i - 1]])

    k = 1
    while k < n:
        # Sort by (rank[i], rank[i+k])
        def key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        new_rank = [0] * n
        new_rank[sa[0]] = 0
        for i in range(1, n):
            prev, cur = sa[i - 1], sa[i]
            same = (rank[prev] == rank[cur] and
                    (rank[prev + k] if prev + k < n else -1) ==
                    (rank[cur + k] if cur + k < n else -1))
            new_rank[cur] = new_rank[prev] + (0 if same else 1)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            break       # all ranks distinct — done early
        k <<= 1
    return sa


# --------------------------------------------------------------------------- #
#  LCP Array — Kasai's O(n) algorithm
# --------------------------------------------------------------------------- #

def build_lcp_array(s: str, sa: list[int]) -> list[int]:
    """
    Returns the LCP array where lcp[i] = length of the longest common prefix
    between suffix sa[i] and suffix sa[i-1] in the sorted order.
    lcp[0] = 0 by convention.

    Example:
        s = "banana", sa = build_suffix_array("banana")
        build_lcp_array("banana", sa)  -> [0, 1, 3, 0, 0, 2]
    """
    n = len(s)
    rank = [0] * n
    for i, v in enumerate(sa):
        rank[v] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h:
                h -= 1
    return lcp


# --------------------------------------------------------------------------- #
#  Application: count distinct substrings
# --------------------------------------------------------------------------- #

def count_distinct_substrings(s: str) -> int:
    """
    Total number of distinct non-empty substrings.
    Formula: n*(n+1)/2 - sum(lcp)

    Example:
        count_distinct_substrings("banana")  -> 15
    """
    n = len(s)
    sa = build_suffix_array(s)
    lcp = build_lcp_array(s, sa)
    return n * (n + 1) // 2 - sum(lcp)


# --------------------------------------------------------------------------- #
#  Application: longest repeated substring
# --------------------------------------------------------------------------- #

def longest_repeated_substring(s: str) -> str:
    """
    Returns the longest substring that appears at least twice.

    Example:
        longest_repeated_substring("banana")  -> "ana"
    """
    sa = build_suffix_array(s)
    lcp = build_lcp_array(s, sa)
    max_lcp = max(lcp)
    if max_lcp == 0:
        return ""
    idx = lcp.index(max_lcp)
    return s[sa[idx]:sa[idx] + max_lcp]


# --------------------------------------------------------------------------- #
#  Application: search for a pattern using binary search on SA
# --------------------------------------------------------------------------- #

def sa_search(s: str, sa: list[int], pattern: str) -> list[int]:
    """
    Returns sorted list of positions where pattern occurs in s.
    Uses binary search: O(m log n).

    Example:
        sa = build_suffix_array("banana")
        sa_search("banana", sa, "ana")  -> [1, 3]
    """
    import bisect
    n, m = len(s), len(pattern)

    lo = bisect.bisect_left(sa, 0,
                             key=lambda i: s[i:i + m] >= pattern)
    hi = bisect.bisect_right(sa, 0,
                              key=lambda i: s[i:i + m] <= pattern)

    # Simpler manual binary search (avoids Python 3.10+ key= for bisect)
    def lower():
        lo_, hi_ = 0, n
        while lo_ < hi_:
            mid = (lo_ + hi_) // 2
            if s[sa[mid]:sa[mid] + m] < pattern:
                lo_ = mid + 1
            else:
                hi_ = mid
        return lo_

    def upper():
        lo_, hi_ = 0, n
        while lo_ < hi_:
            mid = (lo_ + hi_) // 2
            if s[sa[mid]:sa[mid] + m] <= pattern:
                lo_ = mid + 1
            else:
                hi_ = mid
        return lo_

    l, r = lower(), upper()
    return sorted(sa[l:r])


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    s = "banana"
    sa = build_suffix_array(s)
    assert sa == [5, 3, 1, 0, 4, 2], sa

    lcp = build_lcp_array(s, sa)
    assert lcp == [0, 1, 3, 0, 0, 2], lcp

    assert count_distinct_substrings("banana") == 15
    assert longest_repeated_substring("banana") == "ana"

    assert sa_search("banana", sa, "ana") == [1, 3]
    assert sa_search("banana", sa, "nan") == [2]
    assert sa_search("banana", sa, "xyz") == []

    print("All Suffix Array tests passed.")
