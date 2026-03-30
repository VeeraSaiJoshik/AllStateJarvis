"""
NAME: Anagram / Permutation Search with Sliding Window
TAGS: string, sliding window, frequency count, anagram, permutation
DESCRIPTION: Uses a fixed-size sliding window with character frequency tracking to find
             all starting positions where a permutation of pattern appears in text in
             O(n + m) time. Also covers grouped anagram detection, character frequency
             tricks, and the minimum window substring problem.
             Ideal for any competition problem involving character-count invariants.
COMPLEXITY: Time O(n + m), Space O(ALPHA)  — ALPHA = alphabet size (26 for lowercase)
"""

from collections import Counter, defaultdict

# --------------------------------------------------------------------------- #
#  Find all anagram positions in text
# --------------------------------------------------------------------------- #

def find_anagrams(text: str, pattern: str) -> list[int]:
    """
    Returns all start indices i such that text[i:i+len(pattern)] is an anagram
    of pattern.  (i.e. a permutation of pattern)

    Example:
        find_anagrams("cbaebabacd", "abc")  -> [0, 6]
        find_anagrams("abab", "ab")          -> [0, 1, 2]
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []

    need = Counter(pattern)         # required frequencies
    have = Counter(text[:m])        # current window frequencies
    satisfied = sum(1 for c in need if have[c] == need[c])
    total_needed = len(need)

    results = []
    if satisfied == total_needed:
        results.append(0)

    for i in range(m, n):
        # Add right character
        c_in = text[i]
        if c_in in need:
            if have[c_in] == need[c_in] - 1:
                satisfied += 1
        have[c_in] += 1

        # Remove left character
        c_out = text[i - m]
        if c_out in need:
            if have[c_out] == need[c_out]:
                satisfied -= 1
        have[c_out] -= 1

        if satisfied == total_needed:
            results.append(i - m + 1)

    return results


# --------------------------------------------------------------------------- #
#  Check if two strings are anagrams
# --------------------------------------------------------------------------- #

def are_anagrams(s: str, t: str) -> bool:
    """
    Example:
        are_anagrams("listen", "silent")  -> True
        are_anagrams("hello", "world")    -> False
    """
    return Counter(s) == Counter(t)


# --------------------------------------------------------------------------- #
#  Group anagrams together  (classic interview / competition problem)
# --------------------------------------------------------------------------- #

def group_anagrams(words: list[str]) -> list[list[str]]:
    """
    Groups a list of words into sublists of anagrams.
    Uses sorted string as canonical key.

    Example:
        group_anagrams(["eat","tea","tan","ate","nat","bat"])
        -> [["eat","tea","ate"], ["tan","nat"], ["bat"]]
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for w in words:
        groups[''.join(sorted(w))].append(w)
    return list(groups.values())


# --------------------------------------------------------------------------- #
#  Minimum Window Substring
# --------------------------------------------------------------------------- #

def min_window(s: str, t: str) -> str:
    """
    Returns the minimum length substring of s that contains all characters of t
    (with multiplicity).  Returns "" if no such window exists.

    Example:
        min_window("ADOBECODEBANC", "ABC")  -> "BANC"
        min_window("a", "a")               -> "a"
        min_window("a", "b")               -> ""
    """
    if not t or not s:
        return ""
    need = Counter(t)
    have: dict[str, int] = defaultdict(int)
    satisfied = 0
    required = len(need)
    best_start = -1
    best_len = float('inf')
    l = 0

    for r, c in enumerate(s):
        have[c] += 1
        if c in need and have[c] == need[c]:
            satisfied += 1
        while satisfied == required:
            if r - l + 1 < best_len:
                best_len = r - l + 1
                best_start = l
            # Shrink from left
            left_c = s[l]
            have[left_c] -= 1
            if left_c in need and have[left_c] < need[left_c]:
                satisfied -= 1
            l += 1

    return "" if best_start == -1 else s[best_start:best_start + best_len]


# --------------------------------------------------------------------------- #
#  Permutation in String  (binary answer version)
# --------------------------------------------------------------------------- #

def contains_permutation(s: str, pattern: str) -> bool:
    """
    Returns True if any permutation of pattern is a substring of s.

    Example:
        contains_permutation("eidbaooo", "ab")   -> True
        contains_permutation("eidboaoo", "ab")   -> False
    """
    return bool(find_anagrams(s, pattern))


# --------------------------------------------------------------------------- #
#  Smallest window containing all distinct characters of s  (no repeats)
# --------------------------------------------------------------------------- #

def smallest_window_all_distinct(s: str) -> str:
    """
    Returns the shortest substring of s that contains all distinct characters
    of s at least once.

    Example:
        smallest_window_all_distinct("aabcbcdbca")  -> "dbca"
    """
    distinct = set(s)
    return min_window(s, ''.join(distinct))


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    assert find_anagrams("cbaebabacd", "abc") == [0, 6]
    assert find_anagrams("abab", "ab") == [0, 1, 2]
    assert find_anagrams("hello", "xyz") == []

    assert are_anagrams("listen", "silent") is True
    assert are_anagrams("hello", "world") is False

    groups = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    assert len(groups) == 3

    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "b") == ""

    assert contains_permutation("eidbaooo", "ab") is True
    assert contains_permutation("eidboaoo", "ab") is False

    print("All Anagram Search tests passed.")
