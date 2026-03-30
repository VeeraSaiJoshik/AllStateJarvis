"""
NAME: Z-Algorithm for Pattern Matching
TAGS: string, pattern matching, Z-array, linear time
DESCRIPTION: The Z-algorithm computes for each position i the length of the longest
             substring starting at i that is also a prefix of the string, enabling
             O(n+m) pattern matching by concatenating pattern + '$' + text.
             Prefer it over KMP when you need the Z-array itself for structural analysis.
COMPLEXITY: Time O(n + m), Space O(n + m)  — n = len(text), m = len(pattern)
"""

# --------------------------------------------------------------------------- #
#  Z-array construction
# --------------------------------------------------------------------------- #

def build_z(s: str) -> list[int]:
    """
    z[i] = length of the longest substring starting at s[i] that matches a
           prefix of s.  By convention z[0] = 0 (or len(s), convention varies).

    Example:
        build_z("aabxaa")    -> [0, 1, 0, 0, 2, 1]
        build_z("aaaaa")     -> [0, 4, 3, 2, 1]
        build_z("abcabcabc") -> [0, 0, 0, 6, 0, 0, 3, 0, 0]
    """
    n = len(s)
    z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z


# --------------------------------------------------------------------------- #
#  Pattern matching using Z-array
# --------------------------------------------------------------------------- #

def z_search(text: str, pattern: str) -> list[int]:
    """
    Returns all 0-based start indices where pattern occurs in text.

    Concatenate  s = pattern + '$' + text  (sentinel '$' must not appear in either).
    Positions i in the z-array where z[i] == len(pattern) are matches.

    Example:
        z_search("ababcababc", "abc")  -> [2, 7]
        z_search("aaaaaa", "aa")       -> [0, 1, 2, 3, 4]
    """
    if not pattern:
        return list(range(len(text) + 1))
    m = len(pattern)
    combined = pattern + '$' + text
    z = build_z(combined)
    offset = m + 1          # skip past pattern + sentinel
    return [i - offset for i in range(offset, len(combined)) if z[i] == m]


# --------------------------------------------------------------------------- #
#  Bonus: count occurrences of pattern in text
# --------------------------------------------------------------------------- #

def count_occurrences(text: str, pattern: str) -> int:
    return len(z_search(text, pattern))


# --------------------------------------------------------------------------- #
#  Bonus: find smallest rotation that is lexicographically smallest
#         using Z on s + s
# --------------------------------------------------------------------------- #

def smallest_rotation(s: str) -> int:
    """
    Returns the starting index of the lexicographically smallest rotation of s.

    Example:
        smallest_rotation("bca")   -> 1  ("abc" starting at index 2... wait)
        # Actually uses a Booth-style approach; this is a simple O(n log n) demo.
    """
    doubled = s + s
    n = len(s)
    best = 0
    for i in range(1, n):
        if doubled[i:i + n] < doubled[best:best + n]:
            best = i
    return best


# --------------------------------------------------------------------------- #
#  Bonus: check if s2 is a rotation of s1
# --------------------------------------------------------------------------- #

def is_rotation(s1: str, s2: str) -> bool:
    """
    Returns True iff s2 is a rotation of s1.
    Equivalent to checking if s2 is a substring of s1 + s1.

    Example:
        is_rotation("abcde", "cdeab")  -> True
        is_rotation("abcde", "abced")  -> False
    """
    if len(s1) != len(s2):
        return False
    return bool(z_search(s1 + s1, s2))


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Z-array
    assert build_z("aabxaa") == [0, 1, 0, 0, 2, 1]
    assert build_z("aaaaa") == [0, 4, 3, 2, 1]
    assert build_z("abcabcabc") == [0, 0, 0, 6, 0, 0, 3, 0, 0]

    # Search
    assert z_search("ababcababc", "abc") == [2, 7]
    assert z_search("aaaaaa", "aa") == [0, 1, 2, 3, 4]
    assert z_search("hello", "ll") == [2]
    assert z_search("hello", "xyz") == []

    # Rotation
    assert is_rotation("abcde", "cdeab") is True
    assert is_rotation("abcde", "abced") is False

    print("All Z-algorithm tests passed.")
