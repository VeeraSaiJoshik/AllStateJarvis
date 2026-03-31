"""
NAME: Merge Sort, Inversion Count, Quicksort, Quickselect
TAGS: sorting, divide-and-conquer, inversions, k-th smallest
DESCRIPTION: Merge sort with inversion counting (O(n log n)), standard quicksort,
             and quickselect for O(n) average k-th smallest.
             Use inversion count for permutation problems; use quickselect when
             you need k-th order statistic without full sort.
COMPLEXITY: Merge sort O(n log n) time O(n) space; Quickselect O(n) avg O(n^2) worst
"""

from typing import List
import random


# ─── Merge Sort ───────────────────────────────────────────────────────────────

def merge_sort(arr: List[int]) -> List[int]:
    """Standard merge sort. Returns sorted copy."""
    if len(arr) <= 1:
        return arr[:]

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Example:
# merge_sort([5, 2, 4, 1, 3]) -> [1, 2, 3, 4, 5]


# ─── Merge Sort In-Place ──────────────────────────────────────────────────────

def merge_sort_inplace(arr: List[int], lo: int = 0, hi: int = -1) -> None:
    """Sorts arr[lo..hi] in place."""
    if hi == -1:
        hi = len(arr) - 1
    if lo >= hi:
        return
    mid = (lo + hi) // 2
    merge_sort_inplace(arr, lo, mid)
    merge_sort_inplace(arr, mid + 1, hi)
    _merge_inplace(arr, lo, mid, hi)


def _merge_inplace(arr: List[int], lo: int, mid: int, hi: int) -> None:
    temp = arr[lo:hi + 1]
    i, j = 0, mid - lo + 1
    for k in range(lo, hi + 1):
        if i > mid - lo:
            arr[k] = temp[j]; j += 1
        elif j > hi - lo:
            arr[k] = temp[i]; i += 1
        elif temp[i] <= temp[j]:
            arr[k] = temp[i]; i += 1
        else:
            arr[k] = temp[j]; j += 1


# ─── Inversion Count ─────────────────────────────────────────────────────────
# An inversion is a pair (i, j) where i < j but arr[i] > arr[j].
# Count is accumulated during merge: when right[j] < left[i], all elements
# from left[i..end] are inversions with right[j].

def count_inversions(arr: List[int]) -> int:
    """Returns the number of inversions in arr."""
    _, count = _merge_count(arr)
    return count


def _merge_count(arr: List[int]):
    if len(arr) <= 1:
        return arr[:], 0

    mid = len(arr) // 2
    left, lc = _merge_count(arr[:mid])
    right, rc = _merge_count(arr[mid:])
    merged, mc = _merge_and_count(left, right)
    return merged, lc + rc + mc


def _merge_and_count(left: List[int], right: List[int]):
    result = []
    inversions = 0
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            # left[i..] are all > right[j], so they form inversions
            inversions += len(left) - i
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result, inversions

# Example:
# arr = [5, 4, 3, 2, 1] -> 10 inversions (every pair)
# arr = [2, 4, 1, 3, 5] -> 3 inversions: (2,1),(4,1),(4,3)


# ─── Count Reverse Pairs (Leetcode 493) ───────────────────────────────────────
# Count pairs (i, j) where i < j and nums[i] > 2 * nums[j].

def count_reverse_pairs(nums: List[int]) -> int:
    _, count = _rp_sort(nums)
    return count


def _rp_sort(arr: List[int]):
    if len(arr) <= 1:
        return arr[:], 0

    mid = len(arr) // 2
    left, lc = _rp_sort(arr[:mid])
    right, rc = _rp_sort(arr[mid:])

    # Count pairs across left and right
    pairs = 0
    j = 0
    for val in left:
        while j < len(right) and val > 2 * right[j]:
            j += 1
        pairs += j

    return _merge(left, right), lc + rc + pairs

# Example:
# nums = [1, 3, 2, 3, 1] -> 2
# nums = [2, 4, 3, 5, 1] -> 3


# ─── Quicksort ────────────────────────────────────────────────────────────────
# Randomized pivot for O(n log n) expected time.

def quicksort(arr: List[int], lo: int = 0, hi: int = -1) -> None:
    """Sorts arr in place using randomized quicksort."""
    if hi == -1:
        hi = len(arr) - 1
    if lo < hi:
        p = _partition(arr, lo, hi)
        quicksort(arr, lo, p - 1)
        quicksort(arr, p + 1, hi)


def _partition(arr: List[int], lo: int, hi: int) -> int:
    # Randomize pivot to avoid worst-case O(n^2)
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    pivot = arr[hi]

    i = lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1


# ─── 3-Way Quicksort (Dutch National Flag) ───────────────────────────────────
# Efficient when there are many duplicate values.

def quicksort_3way(arr: List[int], lo: int = 0, hi: int = -1) -> None:
    if hi == -1:
        hi = len(arr) - 1
    if lo >= hi:
        return

    # 3-way partition: arr[lo..lt-1] < pivot, arr[lt..gt] == pivot, arr[gt+1..hi] > pivot
    pivot_idx = random.randint(lo, hi)
    arr[lo], arr[pivot_idx] = arr[pivot_idx], arr[lo]
    pivot = arr[lo]

    lt, gt, i = lo, hi, lo + 1
    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1; i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    quicksort_3way(arr, lo, lt - 1)
    quicksort_3way(arr, gt + 1, hi)


# ─── Quickselect: K-th Smallest Element ───────────────────────────────────────
# Average O(n), worst case O(n^2). Use when k-th order statistic is needed.
# For guaranteed O(n), use Median of Medians (not shown here for brevity).

def quickselect(arr: List[int], k: int) -> int:
    """
    Returns the k-th smallest element (0-indexed k).
    Modifies arr in place. Use a copy if original must be preserved.
    """
    arr = arr[:]  # work on a copy
    lo, hi = 0, len(arr) - 1

    while lo < hi:
        p = _partition(arr, lo, hi)
        if p == k:
            return arr[p]
        elif p < k:
            lo = p + 1
        else:
            hi = p - 1

    return arr[lo]

# Example:
# arr = [3, 1, 4, 1, 5, 9, 2, 6], k = 3 -> 3  (0-indexed: sorted is [1,1,2,3,5,6,9], 4th elem)
# Wait: sorted [1,1,2,3,4,5,6,9], k=3 -> 3


# ─── Merge K Sorted Arrays ────────────────────────────────────────────────────
# Repeatedly merge pairs: O(n log k) total.

def merge_k_sorted(arrays: List[List[int]]) -> List[int]:
    while len(arrays) > 1:
        merged = []
        for i in range(0, len(arrays), 2):
            if i + 1 < len(arrays):
                merged.append(_merge(arrays[i], arrays[i + 1]))
            else:
                merged.append(arrays[i])
        arrays = merged
    return arrays[0] if arrays else []

# Example:
# merge_k_sorted([[1,4,7],[2,5,8],[3,6,9]]) -> [1,2,3,4,5,6,7,8,9]


# ─── Sort Nearly Sorted Array (each element at most k away) ──────────────────
# Use a min-heap of size k+1.

def sort_nearly_sorted(arr: List[int], k: int) -> List[int]:
    import heapq
    heap = arr[:k + 1]
    heapq.heapify(heap)
    result = []

    for i in range(k + 1, len(arr)):
        result.append(heapq.heapreplace(heap, arr[i]))

    while heap:
        result.append(heapq.heappop(heap))

    return result

# Example:
# arr = [6, 5, 3, 2, 8, 10, 9], k = 3 -> [2, 3, 5, 6, 8, 9, 10]


if __name__ == "__main__":
    assert merge_sort([5, 2, 4, 1, 3]) == [1, 2, 3, 4, 5]

    arr = [5, 2, 4, 1, 3]
    merge_sort_inplace(arr)
    assert arr == [1, 2, 3, 4, 5]

    assert count_inversions([5, 4, 3, 2, 1]) == 10
    assert count_inversions([2, 4, 1, 3, 5]) == 3

    assert count_reverse_pairs([1, 3, 2, 3, 1]) == 2

    arr = [5, 2, 4, 1, 3]
    quicksort(arr)
    assert arr == [1, 2, 3, 4, 5]

    arr = [5, 2, 4, 1, 3, 3, 3]
    quicksort_3way(arr)
    assert arr == [1, 2, 3, 3, 3, 4, 5]

    assert quickselect([3, 1, 4, 1, 5, 9, 2, 6], 3) == 3

    assert merge_k_sorted([[1,4,7],[2,5,8],[3,6,9]]) == [1,2,3,4,5,6,7,8,9]
    assert sort_nearly_sorted([6, 5, 3, 2, 8, 10, 9], 3) == [2, 3, 5, 6, 8, 9, 10]
    print("All tests passed.")
