"""
NAME: Sliding Window
TAGS: sliding window, string, array, hash map
DESCRIPTION: Maintain a window [lo, hi] over a sequence, expanding the right
             boundary and shrinking the left to satisfy a constraint.
             Use for substring/subarray problems involving counts, sums, or character frequencies.
COMPLEXITY: Time O(n), Space O(1) for fixed alphabet or O(k) for general
"""

from collections import defaultdict
from typing import List


# ─── Fixed-Size Window: Maximum Sum Subarray of Size K ───────────────────────

def max_sum_subarray_k(arr: List[int], k: int) -> int:
    """Maximum sum among all contiguous subarrays of exactly size k."""
    if len(arr) < k:
        return 0

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Example:
# arr = [2, 1, 5, 1, 3, 2], k = 3
# answer -> 9 (subarray [5, 1, 3])


# ─── Fixed-Size Window: Max Average Subarray ─────────────────────────────────

def find_max_average(nums: List[int], k: int) -> float:
    window_sum = sum(nums[:k])
    best = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        best = max(best, window_sum)
    return best / k

# Example:
# nums = [1, 12, -5, -6, 50, 3], k = 4
# answer -> 12.75


# ─── Variable Window: Longest Substring with K Distinct Characters ────────────

def longest_substring_k_distinct(s: str, k: int) -> int:
    """Longest substring containing at most k distinct characters."""
    freq = defaultdict(int)
    lo = 0
    max_len = 0

    for hi in range(len(s)):
        freq[s[hi]] += 1

        # Shrink window until at most k distinct chars
        while len(freq) > k:
            freq[s[lo]] -= 1
            if freq[s[lo]] == 0:
                del freq[s[lo]]
            lo += 1

        max_len = max(max_len, hi - lo + 1)

    return max_len

# Example:
# s = "araaci", k = 2 -> 4 ("araa")
# s = "araaci", k = 1 -> 2 ("aa")


# ─── Minimum Window Substring ─────────────────────────────────────────────────
# Find the smallest window in s containing all characters of t.
# Track how many characters in t are "satisfied" (count >= required).

def min_window_substring(s: str, t: str) -> str:
    if not s or not t:
        return ""

    need = defaultdict(int)
    for c in t:
        need[c] += 1

    required = len(need)   # number of unique chars in t that must be in window
    formed = 0             # number of unique chars in t currently satisfied
    window_counts = defaultdict(int)

    lo = 0
    min_len = float('inf')
    result = ""

    for hi in range(len(s)):
        c = s[hi]
        window_counts[c] += 1

        if c in need and window_counts[c] == need[c]:
            formed += 1

        # Try to shrink window while it contains all chars of t
        while formed == required:
            if hi - lo + 1 < min_len:
                min_len = hi - lo + 1
                result = s[lo:hi + 1]

            window_counts[s[lo]] -= 1
            if s[lo] in need and window_counts[s[lo]] < need[s[lo]]:
                formed -= 1
            lo += 1

    return result

# Example:
# s = "ADOBECODEBANC", t = "ABC" -> "BANC"
# s = "a", t = "a"               -> "a"


# ─── Longest Substring Without Repeating Characters ──────────────────────────

def length_of_longest_substring(s: str) -> int:
    last_seen = {}
    lo = 0
    max_len = 0

    for hi, c in enumerate(s):
        if c in last_seen and last_seen[c] >= lo:
            lo = last_seen[c] + 1
        last_seen[c] = hi
        max_len = max(max_len, hi - lo + 1)

    return max_len

# Example:
# s = "abcabcbb" -> 3 ("abc")
# s = "bbbbb"    -> 1 ("b")
# s = "pwwkew"   -> 3 ("wke")


# ─── Longest Substring with At Most 2 Distinct Characters ────────────────────
# Special case of k-distinct, useful as interview example.

def length_of_longest_substring_two_distinct(s: str) -> int:
    return longest_substring_k_distinct(s, 2)


# ─── Longest Repeating Character Replacement ──────────────────────────────────
# Find the longest substring where you can replace at most k characters
# to make all characters the same.
# Key insight: window is valid if (window_size - max_freq_in_window) <= k.

def character_replacement(s: str, k: int) -> int:
    freq = defaultdict(int)
    lo = 0
    max_freq = 0
    max_len = 0

    for hi in range(len(s)):
        freq[s[hi]] += 1
        max_freq = max(max_freq, freq[s[hi]])

        # If replacements needed exceed k, shrink
        # Note: max_freq may be stale but it only grows, so this is safe
        while (hi - lo + 1) - max_freq > k:
            freq[s[lo]] -= 1
            lo += 1

        max_len = max(max_len, hi - lo + 1)

    return max_len

# Example:
# s = "AABABBA", k = 1 -> 4


# ─── Permutation in String ────────────────────────────────────────────────────
# Check if any permutation of p exists as a substring in s.
# Use fixed-size window of length len(p), match character counts.

def check_inclusion(s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
        return False

    need = defaultdict(int)
    window = defaultdict(int)
    for c in s1:
        need[c] += 1

    k = len(s1)
    for i in range(len(s2)):
        window[s2[i]] += 1
        if i >= k:
            c = s2[i - k]
            window[c] -= 1
            if window[c] == 0:
                del window[c]
        if i >= k - 1 and window == need:
            return True

    return False

# Example:
# s1 = "ab", s2 = "eidbaooo" -> True ("ba" is a permutation)
# s1 = "ab", s2 = "eidboaoo" -> False


# ─── Find All Anagrams in a String ────────────────────────────────────────────
# Return all starting indices where an anagram of p starts in s.

def find_anagrams(s: str, p: str) -> List[int]:
    result = []
    if len(p) > len(s):
        return result

    need = defaultdict(int)
    window = defaultdict(int)
    for c in p:
        need[c] += 1

    k = len(p)
    for i in range(len(s)):
        window[s[i]] += 1
        if i >= k:
            c = s[i - k]
            window[c] -= 1
            if window[c] == 0:
                del window[c]
        if i >= k - 1 and window == need:
            result.append(i - k + 1)

    return result

# Example:
# s = "cbaebabacd", p = "abc" -> [0, 6]


# ─── Max Consecutive Ones III ─────────────────────────────────────────────────
# Flip at most k zeros. Find the longest subarray of ones.

def longest_ones(nums: List[int], k: int) -> int:
    lo = 0
    zeros = 0
    max_len = 0

    for hi in range(len(nums)):
        if nums[hi] == 0:
            zeros += 1
        while zeros > k:
            if nums[lo] == 0:
                zeros -= 1
            lo += 1
        max_len = max(max_len, hi - lo + 1)

    return max_len

# Example:
# nums = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], k = 2
# answer -> 6 (flip positions 5 and 10)


# ─── Fruit Into Baskets (at most 2 types) ─────────────────────────────────────
# Same as longest subarray with at most 2 distinct values.

def total_fruit(fruits: List[int]) -> int:
    return longest_substring_k_distinct(''.join(chr(f) for f in fruits), 2)


# ─── Number of Subarrays with K Distinct Integers ─────────────────────────────
# Exactly k = at-most-k minus at-most-(k-1).

def subarrays_with_k_distinct(nums: List[int], k: int) -> int:
    def at_most_k(k: int) -> int:
        freq = defaultdict(int)
        lo = 0
        count = 0
        for hi in range(len(nums)):
            freq[nums[hi]] += 1
            while len(freq) > k:
                freq[nums[lo]] -= 1
                if freq[nums[lo]] == 0:
                    del freq[nums[lo]]
                lo += 1
            count += hi - lo + 1
        return count

    return at_most_k(k) - at_most_k(k - 1)

# Example:
# nums = [1, 2, 1, 2, 3], k = 2 -> 7


if __name__ == "__main__":
    assert max_sum_subarray_k([2, 1, 5, 1, 3, 2], 3) == 9
    assert longest_substring_k_distinct("araaci", 2) == 4
    assert min_window_substring("ADOBECODEBANC", "ABC") == "BANC"
    assert length_of_longest_substring("abcabcbb") == 3
    assert character_replacement("AABABBA", 1) == 4
    assert check_inclusion("ab", "eidbaooo") == True
    assert find_anagrams("cbaebabacd", "abc") == [0, 6]
    assert longest_ones([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2) == 6
    assert subarrays_with_k_distinct([1, 2, 1, 2, 3], 2) == 7
    print("All tests passed.")
