"""
NAME: Polynomial String Hashing (Single and Double)
TAGS: string, hashing, rolling hash, double hashing, O(1) substring comparison
DESCRIPTION: Precomputes prefix polynomial hashes so any substring hash is retrieved
             in O(1), enabling O(n) duplicate detection, O(n log n) longest common
             extension queries, and binary-search-based longest common substring.
             Use double hashing (two independent mod/base pairs) to drive collision
             probability below 1/10^18.
COMPLEXITY: Time O(n) build, O(1) query; Space O(n)
"""

import random

# --------------------------------------------------------------------------- #
#  Constants  — randomise bases per-run to defeat hash-hack attacks
# --------------------------------------------------------------------------- #

MOD1 = (1 << 61) - 1       # Mersenne prime
MOD2 = (1 << 31) - 1       # second Mersenne prime

# Pick random bases in [256, MOD-1] once at import time
_BASE1 = random.randint(256, MOD1 - 1)
_BASE2 = random.randint(256, MOD2 - 1)


# --------------------------------------------------------------------------- #
#  Single-hash prefix array
# --------------------------------------------------------------------------- #

class StringHash:
    """
    Precomputes prefix hashes for a single (base, mod) pair.

    Usage:
        h = StringHash("abcabc")
        h.get(0, 2)   -> hash of "abc"  (indices 0..2 inclusive)
        h.get(3, 5)   -> hash of "abc"  (same value)
    """

    def __init__(self, s: str, base: int = _BASE1, mod: int = MOD1):
        self.mod = mod
        self.base = base
        n = len(s)
        self.h = [0] * (n + 1)
        self.pw = [1] * (n + 1)
        for i in range(n):
            self.h[i + 1] = (self.h[i] * base + ord(s[i])) % mod
            self.pw[i + 1] = self.pw[i] * base % mod

    def get(self, l: int, r: int) -> int:
        """Hash of s[l..r] inclusive (0-based)."""
        return (self.h[r + 1] - self.h[l] * self.pw[r - l + 1]) % self.mod


# --------------------------------------------------------------------------- #
#  Double-hash prefix array  (recommended for competition use)
# --------------------------------------------------------------------------- #

class DoubleHash:
    """
    Stores two independent hashes per substring for near-zero collision probability.

    Usage:
        dh = DoubleHash("abcabc")
        dh.get(0, 2) == dh.get(3, 5)   -> True  (both are "abc")
        dh.get(0, 2) == dh.get(0, 1)   -> False
    """

    def __init__(self, s: str):
        self._h1 = StringHash(s, _BASE1, MOD1)
        self._h2 = StringHash(s, _BASE2, MOD2)

    def get(self, l: int, r: int) -> tuple[int, int]:
        """Returns (hash1, hash2) for s[l..r] inclusive."""
        return self._h1.get(l, r), self._h2.get(l, r)

    def __len__(self):
        return len(self._h1.h) - 1


# --------------------------------------------------------------------------- #
#  Longest Common Extension (LCE) via binary search + hashing — O(n log n) total
# --------------------------------------------------------------------------- #

def lce(dh1: DoubleHash, i: int, dh2: DoubleHash, j: int) -> int:
    """
    Returns the length of the longest common prefix of
    s1[i:] and s2[j:]  where dh1 is built on s1 and dh2 on s2.

    Example:
        s1, s2 = "abcabc", "abcxyz"
        dh1, dh2 = DoubleHash(s1), DoubleHash(s2)
        lce(dh1, 0, dh2, 0)  -> 3  ("abc" matches)
    """
    n1, n2 = len(dh1), len(dh2)
    max_len = min(n1 - i, n2 - j)
    lo, hi = 0, max_len
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if dh1.get(i, i + mid - 1) == dh2.get(j, j + mid - 1):
            lo = mid
        else:
            hi = mid - 1
    return lo


# --------------------------------------------------------------------------- #
#  Longest Common Substring of two strings — O(n log n)
# --------------------------------------------------------------------------- #

def longest_common_substring(s1: str, s2: str) -> str:
    """
    Returns the longest substring shared by s1 and s2.
    Uses binary search on length + hashing.

    Example:
        longest_common_substring("abcde", "bcdef")  -> "bcde"
        longest_common_substring("xyz", "abc")       -> ""
    """
    dh1 = DoubleHash(s1)
    dh2 = DoubleHash(s2)
    n1, n2 = len(s1), len(s2)

    def has_common_of_length(length: int) -> tuple[int, int] | None:
        seen: set[tuple[int, int]] = set()
        for i in range(n1 - length + 1):
            seen.add(dh1.get(i, i + length - 1))
        for j in range(n2 - length + 1):
            h = dh2.get(j, j + length - 1)
            if h in seen:
                return j, length
        return None

    lo, hi = 0, min(n1, n2)
    best_j, best_len = 0, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        found = has_common_of_length(mid)
        if found:
            best_j, best_len = found
            lo = mid + 1
        else:
            hi = mid - 1
    return s2[best_j:best_j + best_len]


# --------------------------------------------------------------------------- #
#  Find all duplicate substrings of length k  — O(n)
# --------------------------------------------------------------------------- #

def find_duplicate_substrings(s: str, k: int) -> list[str]:
    """
    Returns all distinct substrings of length k that appear more than once.

    Example:
        find_duplicate_substrings("abcabc", 3)  -> ["abc"]
        find_duplicate_substrings("aaaa", 2)    -> ["aa"]
    """
    dh = DoubleHash(s)
    n = len(s)
    if k > n:
        return []
    seen: dict[tuple[int, int], int] = {}
    dupes: set[tuple[int, int]] = set()
    for i in range(n - k + 1):
        h = dh.get(i, i + k - 1)
        if h in seen:
            dupes.add(h)
        else:
            seen[h] = i
    return [s[seen[h]:seen[h] + k] for h in dupes]


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    dh = DoubleHash("abcabc")
    assert dh.get(0, 2) == dh.get(3, 5)    # "abc" == "abc"
    assert dh.get(0, 2) != dh.get(0, 1)    # "abc" != "ab"

    s1, s2 = "abcabc", "abcxyz"
    dh1, dh2 = DoubleHash(s1), DoubleHash(s2)
    assert lce(dh1, 0, dh2, 0) == 3

    assert longest_common_substring("abcde", "bcdef") == "bcde"
    assert longest_common_substring("xyz", "abc") == ""
    assert longest_common_substring("abab", "baba") in ("aba", "bab", "ab", "ba")

    dups = find_duplicate_substrings("abcabc", 3)
    assert dups == ["abc"]

    dups2 = find_duplicate_substrings("aaaa", 2)
    assert dups2 == ["aa"]

    print("All String Hashing tests passed.")
