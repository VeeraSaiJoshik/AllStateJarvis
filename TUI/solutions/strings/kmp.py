"""
NAME: KMP (Knuth-Morris-Pratt) Pattern Matching
TAGS: string, pattern matching, failure function, linear time
DESCRIPTION: KMP finds all occurrences of a pattern in a text in O(n+m) time using a
             precomputed failure (partial match) function that avoids redundant comparisons.
             Use in competitions when you need efficient single-pattern matching or need
             to detect repeated string structure.
COMPLEXITY: Time O(n + m), Space O(m)  — n = len(text), m = len(pattern)
"""

# --------------------------------------------------------------------------- #
#  Failure function (also called pi / partial-match / lps array)
# --------------------------------------------------------------------------- #

def build_failure(pattern: str) -> list[int]:
    """
    failure[i] = length of the longest proper prefix of pattern[:i+1]
                 that is also a suffix.
    """
    m = len(pattern)
    fail = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = fail[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        fail[i] = j
    return fail


# --------------------------------------------------------------------------- #
#  KMP search — returns list of starting indices (0-based) of all occurrences
# --------------------------------------------------------------------------- #

def kmp_search(text: str, pattern: str) -> list[int]:
    """
    Returns all 0-based start positions where pattern appears in text.

    Example:
        kmp_search("ababcababc", "abc")  -> [2, 7]
        kmp_search("aaaaaa", "aa")       -> [0, 1, 2, 3, 4]
    """
    if not pattern:
        return list(range(len(text) + 1))
    n, m = len(text), len(pattern)
    fail = build_failure(pattern)
    results = []
    j = 0                       # matched length so far
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = fail[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            results.append(i - m + 1)
            j = fail[j - 1]    # continue searching for overlapping matches
    return results


# --------------------------------------------------------------------------- #
#  Bonus: period / smallest repeating unit of a string
# --------------------------------------------------------------------------- #

def smallest_period(s: str) -> int:
    """
    Returns the length of the smallest period p such that s is a prefix of (s[:p] * k).
    Uses the failure function: period = len(s) - fail[-1].

    Example:
        smallest_period("abababab")  -> 2
        smallest_period("abcabc")   -> 3
        smallest_period("abcd")     -> 4  (no repetition)
    """
    if not s:
        return 0
    fail = build_failure(s)
    n = len(s)
    period = n - fail[-1]
    # True period only if n % period == 0; otherwise the string is not fully periodic
    return period if n % period == 0 else n


# --------------------------------------------------------------------------- #
#  Bonus: count distinct substrings that match pattern (overlapping allowed)
# --------------------------------------------------------------------------- #

def count_occurrences(text: str, pattern: str) -> int:
    return len(kmp_search(text, pattern))


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Basic search
    assert kmp_search("ababcababc", "abc") == [2, 7]
    assert kmp_search("aaaaaa", "aa") == [0, 1, 2, 3, 4]
    assert kmp_search("hello", "ll") == [2]
    assert kmp_search("hello", "xyz") == []

    # Failure function
    assert build_failure("ababab") == [0, 0, 1, 2, 3, 4]
    assert build_failure("aabaabaab") == [0, 1, 0, 1, 2, 3, 4, 5, 6]

    # Period
    assert smallest_period("abababab") == 2
    assert smallest_period("abcabc") == 3
    assert smallest_period("abcd") == 4

    print("All KMP tests passed.")
