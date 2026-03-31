"""
NAME: Rabin-Karp Rolling Hash for Pattern Matching
TAGS: string, hashing, rolling hash, multi-pattern, polynomial hashing
DESCRIPTION: Rabin-Karp uses polynomial rolling hashes to find pattern occurrences in
             O(n + m) expected time, and trivially extends to multi-pattern matching
             by hashing all patterns into a set. Use when you need to match many short
             patterns simultaneously or when implementing substring equality checks.
COMPLEXITY: Time O(n + m) expected, O(nm) worst case; Space O(1) single / O(k) multi-pattern
"""

MOD1 = (1 << 61) - 1          # Mersenne prime — very fast modular arithmetic
MOD2 = (1 << 31) - 1          # second modulus for double hashing
BASE1 = 131
BASE2 = 137


# --------------------------------------------------------------------------- #
#  Fast modular arithmetic for Mersenne prime 2^61 - 1
# --------------------------------------------------------------------------- #

def _mod61(x: int) -> int:
    x = (x >> 61) + (x & MOD1)
    if x >= MOD1:
        x -= MOD1
    return x


def _mul61(a: int, b: int) -> int:
    return _mod61(a * b)          # Python big-int handles the intermediate product


# --------------------------------------------------------------------------- #
#  Single rolling hash
# --------------------------------------------------------------------------- #

def _hash_single(s: str, base: int, mod: int) -> int:
    h = 0
    for c in s:
        h = (h * base + ord(c)) % mod
    return h


def rabin_karp_search(text: str, pattern: str) -> list[int]:
    """
    Returns all 0-based start indices where pattern appears in text.
    Uses double hashing to reduce collision probability to near zero.

    Example:
        rabin_karp_search("ababcababc", "abc")  -> [2, 7]
        rabin_karp_search("aaaaaa", "aa")        -> [0, 1, 2, 3, 4]
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []
    if m == 0:
        return list(range(n + 1))

    ph1 = _hash_single(pattern, BASE1, MOD1)
    ph2 = _hash_single(pattern, BASE2, MOD2)

    # Precompute highest-order multiplier:  base^(m-1) mod mod
    pow1 = pow(BASE1, m - 1, MOD1)
    pow2 = pow(BASE2, m - 1, MOD2)

    th1 = _hash_single(text[:m], BASE1, MOD1)
    th2 = _hash_single(text[:m], BASE2, MOD2)

    results = []
    for i in range(n - m + 1):
        if th1 == ph1 and th2 == ph2:
            # Verify to avoid false positives
            if text[i:i + m] == pattern:
                results.append(i)
        if i + m < n:
            th1 = (th1 - ord(text[i]) * pow1 % MOD1 + MOD1) * BASE1 % MOD1 + ord(text[i + m])
            th1 %= MOD1
            th2 = (th2 - ord(text[i]) * pow2 % MOD2 + MOD2) * BASE2 % MOD2 + ord(text[i + m])
            th2 %= MOD2
    return results


# --------------------------------------------------------------------------- #
#  Multi-pattern matching  (all patterns of the same length)
# --------------------------------------------------------------------------- #

def rabin_karp_multi(text: str, patterns: list[str]) -> dict[str, list[int]]:
    """
    Finds all occurrences of multiple patterns (all same length) in text.
    Returns a dict mapping each pattern to its list of start indices.

    For patterns of different lengths, group by length and call once per group.

    Example:
        rabin_karp_multi("abcabdabc", ["abc", "abd"])
        -> {"abc": [0, 6], "abd": [3]}
    """
    if not patterns:
        return {}
    m = len(patterns[0])
    # Build hash set of (hash1, hash2) -> pattern
    pat_hashes: dict[tuple[int, int], list[str]] = {}
    for p in patterns:
        if len(p) != m:
            raise ValueError("All patterns must have the same length for this function.")
        key = (_hash_single(p, BASE1, MOD1), _hash_single(p, BASE2, MOD2))
        pat_hashes.setdefault(key, []).append(p)

    result: dict[str, list[int]] = {p: [] for p in patterns}
    n = len(text)
    if m > n:
        return result

    pow1 = pow(BASE1, m - 1, MOD1)
    pow2 = pow(BASE2, m - 1, MOD2)
    th1 = _hash_single(text[:m], BASE1, MOD1)
    th2 = _hash_single(text[:m], BASE2, MOD2)

    for i in range(n - m + 1):
        key = (th1, th2)
        if key in pat_hashes:
            window = text[i:i + m]
            for p in pat_hashes[key]:
                if window == p:
                    result[p].append(i)
        if i + m < n:
            th1 = (th1 - ord(text[i]) * pow1 % MOD1 + MOD1) * BASE1 % MOD1 + ord(text[i + m])
            th1 %= MOD1
            th2 = (th2 - ord(text[i]) * pow2 % MOD2 + MOD2) * BASE2 % MOD2 + ord(text[i + m])
            th2 %= MOD2
    return result


# --------------------------------------------------------------------------- #
#  Utility: hash a single string (useful for building custom solutions)
# --------------------------------------------------------------------------- #

def string_hash(s: str) -> tuple[int, int]:
    """Returns (hash1, hash2) double hash for a string."""
    return _hash_single(s, BASE1, MOD1), _hash_single(s, BASE2, MOD2)


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    assert rabin_karp_search("ababcababc", "abc") == [2, 7]
    assert rabin_karp_search("aaaaaa", "aa") == [0, 1, 2, 3, 4]
    assert rabin_karp_search("hello", "ll") == [2]
    assert rabin_karp_search("hello", "xyz") == []
    assert rabin_karp_search("abc", "abcd") == []

    res = rabin_karp_multi("abcabdabc", ["abc", "abd"])
    assert res["abc"] == [0, 6]
    assert res["abd"] == [3]

    print("All Rabin-Karp tests passed.")
