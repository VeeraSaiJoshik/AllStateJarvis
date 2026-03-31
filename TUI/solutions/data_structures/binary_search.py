"""
NAME: Binary Search Templates
TAGS: binary search, array, search, divide-and-conquer
DESCRIPTION: Binary search templates for exact match, first/last occurrence,
             search on answer (parametric search), and rotated/sorted arrays.
             Reduces O(n) linear scan to O(log n); use whenever the search space is monotone.
COMPLEXITY: Time O(log n), Space O(1)
"""

from typing import List, Callable
import math


# ─── Template 1: Exact Match ──────────────────────────────────────────────────
# Standard binary search. Returns index or -1.

def binary_search(arr: List[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# Example:
# arr = [1, 3, 5, 7, 9], target = 5 -> 2


# ─── Template 2: First Occurrence (Lower Bound) ───────────────────────────────
# Returns the leftmost index where arr[i] >= target (bisect_left equivalent).

def lower_bound(arr: List[int], target: int) -> int:
    """Index of first element >= target. Returns len(arr) if all elements < target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# Example:
# arr = [1, 2, 2, 4, 5], target = 2 -> 1 (first 2)
# arr = [1, 2, 2, 4, 5], target = 3 -> 3 (insertion point)


# ─── Template 3: Last Occurrence (Upper Bound) ────────────────────────────────
# Returns the leftmost index where arr[i] > target (bisect_right equivalent).
# last occurrence of target is upper_bound - 1.

def upper_bound(arr: List[int], target: int) -> int:
    """Index of first element > target. Returns len(arr) if all elements <= target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# Example:
# arr = [1, 2, 2, 4, 5], target = 2 -> 3 (one past last 2)


def first_occurrence(arr: List[int], target: int) -> int:
    idx = lower_bound(arr, target)
    return idx if idx < len(arr) and arr[idx] == target else -1

def last_occurrence(arr: List[int], target: int) -> int:
    idx = upper_bound(arr, target) - 1
    return idx if idx >= 0 and arr[idx] == target else -1

def count_occurrences(arr: List[int], target: int) -> int:
    return upper_bound(arr, target) - lower_bound(arr, target)


# ─── Template 4: Search on Answer (Parametric Binary Search) ─────────────────
# When you binary search on the answer rather than an index.
# Pattern: find the minimum/maximum value x such that condition(x) is True.

def binary_search_on_answer(lo: int, hi: int, condition: Callable[[int], bool]) -> int:
    """
    Find the MINIMUM x in [lo, hi] such that condition(x) is True.
    Assumes condition is monotone: False...False True...True.
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if condition(mid):
            hi = mid        # mid could be the answer; don't discard it
        else:
            lo = mid + 1    # mid is not the answer; discard it
    return lo


def binary_search_on_answer_max(lo: int, hi: int, condition: Callable[[int], bool]) -> int:
    """
    Find the MAXIMUM x in [lo, hi] such that condition(x) is True.
    Assumes condition is monotone: True...True False...False.
    """
    while lo < hi:
        mid = (lo + hi + 1) // 2   # upper mid to avoid infinite loop
        if condition(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


# ─── Application: Minimum Pages (Book Allocation Problem) ─────────────────────
# Allocate books to m students such that max pages assigned is minimized.
# Binary search on the answer (max pages).

def min_pages(books: List[int], m: int) -> int:
    if m > len(books):
        return -1

    def can_allocate(max_pages: int) -> bool:
        students = 1
        curr = 0
        for b in books:
            if b > max_pages:
                return False
            if curr + b > max_pages:
                students += 1
                curr = b
            else:
                curr += b
        return students <= m

    lo = max(books)
    hi = sum(books)
    return binary_search_on_answer(lo, hi, can_allocate)

# Example:
# books = [12, 34, 67, 90], m = 2 -> 113


# ─── Application: Koko Eating Bananas ─────────────────────────────────────────

def min_eating_speed(piles: List[int], h: int) -> int:
    def can_finish(speed: int) -> bool:
        return sum(math.ceil(p / speed) for p in piles) <= h

    return binary_search_on_answer(1, max(piles), can_finish)

# Example:
# piles = [3, 6, 7, 11], h = 8 -> 4


# ─── Application: Capacity to Ship Packages ───────────────────────────────────

def ship_within_days(weights: List[int], days: int) -> int:
    def can_ship(capacity: int) -> bool:
        trips = 1
        load = 0
        for w in weights:
            if load + w > capacity:
                trips += 1
                load = 0
            load += w
        return trips <= days

    return binary_search_on_answer(max(weights), sum(weights), can_ship)

# Example:
# weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], days = 5 -> 15


# ─── Template 5: Search in Rotated Sorted Array ───────────────────────────────
# One half is always sorted. Determine which half and check if target is in it.

def search_rotated(nums: List[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1

    return -1

# Example:
# nums = [4, 5, 6, 7, 0, 1, 2], target = 0 -> 4
# nums = [4, 5, 6, 7, 0, 1, 2], target = 3 -> -1


# ─── Find Minimum in Rotated Sorted Array ─────────────────────────────────────

def find_min_rotated(nums: List[int]) -> int:
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1   # min is in right half
        else:
            hi = mid       # mid could be the min
    return nums[lo]

# Example:
# nums = [3, 4, 5, 1, 2] -> 1
# nums = [4, 5, 6, 7, 0, 1, 2] -> 0


# ─── Find Peak Element ────────────────────────────────────────────────────────
# A peak is an element greater than its neighbors. Always exists, find any.

def find_peak_element(nums: List[int]) -> int:
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[mid + 1]:
            hi = mid       # peak is on left side (including mid)
        else:
            lo = mid + 1   # peak is on right side
    return lo

# Example:
# nums = [1, 2, 3, 1] -> 2
# nums = [1, 2, 1, 3, 5, 6, 4] -> 1 or 5


# ─── Sqrt (Integer Square Root) ───────────────────────────────────────────────

def int_sqrt(x: int) -> int:
    """Largest integer k such that k*k <= x."""
    if x < 2:
        return x
    lo, hi = 1, x // 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * mid <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo

# Example:
# x = 8 -> 2  (2*2=4 <= 8, 3*3=9 > 8)


# ─── Fractional / Real-Valued Binary Search ───────────────────────────────────
# When the answer is a real number. Use a fixed number of iterations.

def real_binary_search(lo: float, hi: float,
                        condition: Callable[[float], bool],
                        iterations: int = 100) -> float:
    """
    Find the real-valued x in [lo, hi] where condition transitions.
    Use ~100 iterations for ~30 decimal places of precision.
    """
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if condition(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

# Example: Find sqrt(2) with real binary search
# real_binary_search(1.0, 2.0, lambda x: x*x >= 2) -> 1.41421356...


# ─── K-th Smallest in Sorted Matrix ──────────────────────────────────────────
# Binary search on value. Count elements <= mid using staircase walk.

def kth_smallest_matrix(matrix: List[List[int]], k: int) -> int:
    n = len(matrix)

    def count_less_equal(target: int) -> int:
        count = 0
        row, col = n - 1, 0
        while row >= 0 and col < n:
            if matrix[row][col] <= target:
                count += row + 1
                col += 1
            else:
                row -= 1
        return count

    lo, hi = matrix[0][0], matrix[-1][-1]
    while lo < hi:
        mid = (lo + hi) // 2
        if count_less_equal(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo

# Example:
# matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8 -> 13


# ─── Median of Two Sorted Arrays ──────────────────────────────────────────────
# Binary search on the partition of the smaller array. O(log(min(m,n))).

def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    lo, hi = 0, m

    while lo <= hi:
        partition1 = (lo + hi) // 2
        partition2 = (m + n + 1) // 2 - partition1

        max_left1  = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf')  if partition1 == m else nums1[partition1]
        max_left2  = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf')  if partition2 == n else nums2[partition2]

        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            hi = partition1 - 1
        else:
            lo = partition1 + 1

    raise ValueError("Input arrays are not sorted")

# Example:
# nums1 = [1, 3], nums2 = [2] -> 2.0
# nums1 = [1, 2], nums2 = [3, 4] -> 2.5


if __name__ == "__main__":
    assert binary_search([1, 3, 5, 7, 9], 5) == 2
    assert lower_bound([1, 2, 2, 4, 5], 2) == 1
    assert upper_bound([1, 2, 2, 4, 5], 2) == 3
    assert first_occurrence([1, 2, 2, 4, 5], 2) == 1
    assert last_occurrence([1, 2, 2, 4, 5], 2) == 2
    assert count_occurrences([1, 2, 2, 2, 5], 2) == 3
    assert min_pages([12, 34, 67, 90], 2) == 113
    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert ship_within_days([1,2,3,4,5,6,7,8,9,10], 5) == 15
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert find_min_rotated([3, 4, 5, 1, 2]) == 1
    assert int_sqrt(8) == 2
    assert kth_smallest_matrix([[1,5,9],[10,11,13],[12,13,15]], 8) == 13
    assert find_median_sorted_arrays([1, 3], [2]) == 2.0
    print("All tests passed.")
