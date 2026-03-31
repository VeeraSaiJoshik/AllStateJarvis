"""
NAME: Ordered Set (SortedList / Order Statistics)
TAGS: sorted-list, order-statistics, binary-search, rank-queries
DESCRIPTION: An ordered multiset backed by sortedcontainers.SortedList, supporting
             O(log n) insert/delete and O(log n) rank/order-statistic queries.
             Use in competitions for problems requiring dynamic sorted sequences,
             k-th smallest, or count of elements in a range.
COMPLEXITY: Time O(log n) per operation, Space O(n)
"""

# pip install sortedcontainers  (available in most online judges including Codeforces)
from sortedcontainers import SortedList
from typing import List, Optional


# ─── Basic SortedList Operations ──────────────────────────────────────────────

def demo_sorted_list():
    sl = SortedList()

    # Add elements
    sl.add(3); sl.add(1); sl.add(4); sl.add(1); sl.add(5)
    # sl is now [1, 1, 3, 4, 5]

    # O(log n) membership
    assert 3 in sl
    assert 2 not in sl

    # O(log n) rank (0-indexed position of first occurrence)
    assert sl.bisect_left(3) == 2   # index of first element >= 3
    assert sl.bisect_right(3) == 3  # index of first element > 3

    # O(log n) removal
    sl.remove(1)   # removes one occurrence
    # sl is now [1, 3, 4, 5]
    sl.discard(9)  # no-op if not present

    # O(1) index access
    assert sl[0] == 1   # minimum
    assert sl[-1] == 5  # maximum

    # Count elements in range [lo, hi]
    count_in_range = sl.bisect_right(4) - sl.bisect_left(2)
    # elements in [2, 4]: 3, 4 -> count = 2
    assert count_in_range == 2

    return sl


# ─── K-th Smallest Element (0-indexed) ───────────────────────────────────────
# Direct O(log n) index access.

def kth_smallest(sl: SortedList, k: int) -> int:
    return sl[k]

# Example:
# sl = SortedList([3, 1, 4, 1, 5, 9])
# kth_smallest(sl, 2) -> 3  (0-indexed, elements: [1,1,3,4,5,9])


# ─── Rank of an Element ────────────────────────────────────────────────────────
# How many elements are strictly less than x?

def rank(sl: SortedList, x: int) -> int:
    return sl.bisect_left(x)

# Example:
# sl = SortedList([1, 2, 4, 5])
# rank(sl, 3) -> 2  (elements 1, 2 are less than 3)


# ─── Count Smaller Numbers After Self ────────────────────────────────────────
# For each element, count how many elements to its right are smaller.
# O(n log n) with SortedList.

def count_smaller(nums: List[int]) -> List[int]:
    sl = SortedList()
    result = []

    for num in reversed(nums):
        result.append(sl.bisect_left(num))  # count of elements < num
        sl.add(num)

    return result[::-1]

# Example:
# nums = [5, 2, 6, 1] -> [2, 1, 1, 0]


# ─── Count of Range Sum ───────────────────────────────────────────────────────
# Count subarrays where sum is in [lower, upper].
# Use prefix sums + SortedList for O(n log n).

def count_range_sum(nums: List[int], lower: int, upper: int) -> int:
    sl = SortedList([0])
    prefix = 0
    count = 0

    for num in nums:
        prefix += num
        # We need: lower <= prefix - prev_prefix <= upper
        # i.e., prefix - upper <= prev_prefix <= prefix - lower
        lo = sl.bisect_left(prefix - upper)
        hi = sl.bisect_right(prefix - lower)
        count += hi - lo
        sl.add(prefix)

    return count

# Example:
# nums = [-2, 5, -1], lower = -2, upper = 2 -> 3


# ─── Sliding Window: Count Elements in Range ──────────────────────────────────
# Maintain a SortedList of a sliding window, query rank at each step.

def sliding_window_rank_queries(nums: List[int], k: int, target: int) -> List[int]:
    """
    For each window of size k, return the rank of `target` in that window.
    rank = number of elements strictly less than target.
    """
    sl = SortedList()
    result = []

    for i, num in enumerate(nums):
        sl.add(num)
        if i >= k:
            sl.remove(nums[i - k])
        if i >= k - 1:
            result.append(sl.bisect_left(target))

    return result


# ─── Contains Duplicate Within K Distance and Value Range ────────────────────
# Check if there exist i, j such that |i-j| <= k and |nums[i]-nums[j]| <= t.
# Use SortedList as a sliding window of size k.

def contains_nearby_almost_duplicate(nums: List[int], k: int, t: int) -> bool:
    sl = SortedList()

    for i, num in enumerate(nums):
        # Check if any element in window is in [num - t, num + t]
        pos = sl.bisect_left(num - t)
        if pos < len(sl) and sl[pos] <= num + t:
            return True
        sl.add(num)
        if len(sl) > k:
            sl.remove(nums[i - k])

    return False

# Example:
# nums = [1, 2, 3, 1], k = 3, t = 0 -> True
# nums = [1, 5, 9, 1, 5, 9], k = 2, t = 3 -> False


# ─── Order Statistics Tree (wrapper with rank/select) ────────────────────────
class OrderStatisticsTree:
    """
    A dynamic ordered multiset supporting:
    - insert(x): O(log n)
    - delete(x): O(log n)
    - rank(x): count of elements < x, O(log n)
    - select(k): k-th smallest (0-indexed), O(log n)
    - count_range(lo, hi): count elements in [lo, hi], O(log n)
    """
    def __init__(self):
        self._sl = SortedList()

    def insert(self, x) -> None:
        self._sl.add(x)

    def delete(self, x) -> None:
        self._sl.remove(x)

    def discard(self, x) -> None:
        self._sl.discard(x)

    def rank(self, x) -> int:
        """Number of elements strictly less than x."""
        return self._sl.bisect_left(x)

    def rank_leq(self, x) -> int:
        """Number of elements <= x."""
        return self._sl.bisect_right(x)

    def select(self, k: int):
        """k-th smallest element (0-indexed)."""
        return self._sl[k]

    def count_range(self, lo, hi) -> int:
        """Count elements in [lo, hi] inclusive."""
        return self._sl.bisect_right(hi) - self._sl.bisect_left(lo)

    def min(self):
        return self._sl[0]

    def max(self):
        return self._sl[-1]

    def predecessor(self, x):
        """Largest element strictly less than x."""
        idx = self._sl.bisect_left(x) - 1
        return self._sl[idx] if idx >= 0 else None

    def successor(self, x):
        """Smallest element strictly greater than x."""
        idx = self._sl.bisect_right(x)
        return self._sl[idx] if idx < len(self._sl) else None

    def __len__(self):
        return len(self._sl)

    def __contains__(self, x):
        return x in self._sl

# Example:
# ost = OrderStatisticsTree()
# for v in [5, 2, 8, 1, 9, 3]: ost.insert(v)
# ost.rank(5)   -> 3  (1, 2, 3 are less than 5)
# ost.select(0) -> 1  (smallest)
# ost.select(2) -> 3  (3rd smallest)
# ost.count_range(2, 7) -> 3  (2, 3, 5)
# ost.predecessor(5) -> 3
# ost.successor(5)   -> 8


# ─── Longest Increasing Subsequence Length (patience sorting) ─────────────────
# Classic O(n log n) LIS using SortedList as a BIT-alternative.

def lis_length(nums: List[int]) -> int:
    # Use a plain list for patience sorting tails (SortedList doesn't support item assignment)
    tails = []
    from bisect import bisect_left
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num  # replace to keep tails minimal
    return len(tails)

# Example:
# nums = [10, 9, 2, 5, 3, 7, 101, 18] -> 4 ([2,3,7,18] or [2,5,7,18])


# ─── Merge Two Sorted Arrays (using SortedList as merge buffer) ──────────────
def merge_sorted(a: List[int], b: List[int]) -> List[int]:
    sl = SortedList(a)
    sl.update(b)
    return list(sl)


if __name__ == "__main__":
    demo_sorted_list()

    sl = SortedList([1, 1, 3, 4, 5, 9])
    assert kth_smallest(sl, 2) == 3
    assert rank(sl, 3) == 2

    assert count_smaller([5, 2, 6, 1]) == [2, 1, 1, 0]
    assert count_range_sum([-2, 5, -1], -2, 2) == 3

    assert contains_nearby_almost_duplicate([1, 2, 3, 1], 3, 0) == True
    assert contains_nearby_almost_duplicate([1, 5, 9, 1, 5, 9], 2, 3) == False

    ost = OrderStatisticsTree()
    for v in [5, 2, 8, 1, 9, 3]:
        ost.insert(v)
    assert ost.rank(5) == 3
    assert ost.select(0) == 1
    assert ost.select(2) == 3
    assert ost.count_range(2, 7) == 3
    assert ost.predecessor(5) == 3
    assert ost.successor(5) == 8

    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    print("All tests passed.")
