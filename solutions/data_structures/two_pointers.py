"""
NAME: Two Pointers Technique
TAGS: two-pointers, array, sorting, greedy
DESCRIPTION: Use two indices moving toward each other (or in the same direction)
             to reduce O(n^2) brute-force to O(n) or O(n log n) with sorting.
             Essential for pair/triplet sum problems, container problems, and sorted array queries.
COMPLEXITY: Time O(n) or O(n log n) with sort, Space O(1) or O(n) for results
"""

from typing import List, Tuple


# ─── Pair Sum (Two Sum II - sorted input) ─────────────────────────────────────
# Find a pair that sums to target. Array must be sorted.
# Returns 1-based indices (LeetCode style) or (-1, -1) if not found.

def pair_sum_sorted(arr: List[int], target: int) -> Tuple[int, int]:
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        s = arr[lo] + arr[hi]
        if s == target:
            return (lo + 1, hi + 1)  # 1-indexed
        elif s < target:
            lo += 1
        else:
            hi -= 1
    return (-1, -1)

# Example:
# arr = [2, 7, 11, 15], target = 9 -> (1, 2)


# ─── All Pairs with Given Sum ─────────────────────────────────────────────────
def all_pairs_sum(arr: List[int], target: int) -> List[Tuple[int, int]]:
    arr.sort()
    lo, hi = 0, len(arr) - 1
    result = []
    while lo < hi:
        s = arr[lo] + arr[hi]
        if s == target:
            result.append((arr[lo], arr[hi]))
            # Skip duplicates
            while lo < hi and arr[lo] == arr[lo + 1]:
                lo += 1
            while lo < hi and arr[hi] == arr[hi - 1]:
                hi -= 1
            lo += 1
            hi -= 1
        elif s < target:
            lo += 1
        else:
            hi -= 1
    return result


# ─── 3Sum (all unique triplets summing to zero) ───────────────────────────────
# Fix one element, two-pointer on the rest. Sort first to handle duplicates.

def three_sum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 2):
        # Skip duplicate values for the fixed element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        # Early termination: smallest possible triplet too large
        if nums[i] > 0:
            break

        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s == 0:
                result.append([nums[i], nums[lo], nums[hi]])
                while lo < hi and nums[lo] == nums[lo + 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi - 1]:
                    hi -= 1
                lo += 1
                hi -= 1
            elif s < 0:
                lo += 1
            else:
                hi -= 1

    return result

# Example:
# nums = [-1, 0, 1, 2, -1, -4]
# result -> [[-1, -1, 2], [-1, 0, 1]]


# ─── 3Sum Closest ─────────────────────────────────────────────────────────────
# Find triplet whose sum is closest to target.

def three_sum_closest(nums: List[int], target: int) -> int:
    nums.sort()
    n = len(nums)
    closest = float('inf')

    for i in range(n - 2):
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if abs(s - target) < abs(closest - target):
                closest = s
            if s < target:
                lo += 1
            elif s > target:
                hi -= 1
            else:
                return s  # exact match

    return closest

# Example:
# nums = [-1, 2, 1, -4], target = 1 -> 2


# ─── 4Sum ─────────────────────────────────────────────────────────────────────
# Generalized: reduce to 3Sum by fixing one more element.

def four_sum(nums: List[int], target: int) -> List[List[int]]:
    nums.sort()
    n = len(nums)
    result = []

    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            lo, hi = j + 1, n - 1
            while lo < hi:
                s = nums[i] + nums[j] + nums[lo] + nums[hi]
                if s == target:
                    result.append([nums[i], nums[j], nums[lo], nums[hi]])
                    while lo < hi and nums[lo] == nums[lo + 1]:
                        lo += 1
                    while lo < hi and nums[hi] == nums[hi - 1]:
                        hi -= 1
                    lo += 1
                    hi -= 1
                elif s < target:
                    lo += 1
                else:
                    hi -= 1

    return result


# ─── Container With Most Water ────────────────────────────────────────────────
# Two walls, maximize the water held: area = min(h[l], h[r]) * (r - l).
# Greedy: always move the shorter wall inward.

def max_water_container(height: List[int]) -> int:
    lo, hi = 0, len(height) - 1
    max_area = 0

    while lo < hi:
        area = min(height[lo], height[hi]) * (hi - lo)
        max_area = max(max_area, area)
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1

    return max_area

# Example:
# height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# answer -> 49


# ─── Trapping Rain Water ──────────────────────────────────────────────────────
# Two-pointer approach: track max from left and right.
# Water at i = min(max_left, max_right) - height[i].

def trap_rain_water(height: List[int]) -> int:
    lo, hi = 0, len(height) - 1
    max_left = max_right = 0
    water = 0

    while lo < hi:
        if height[lo] < height[hi]:
            if height[lo] >= max_left:
                max_left = height[lo]
            else:
                water += max_left - height[lo]
            lo += 1
        else:
            if height[hi] >= max_right:
                max_right = height[hi]
            else:
                water += max_right - height[hi]
            hi -= 1

    return water

# Example:
# height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
# answer -> 6


# ─── Remove Duplicates from Sorted Array (in-place) ──────────────────────────
# Slow/fast pointer variant: slow tracks write position.

def remove_duplicates(nums: List[int]) -> int:
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1

# Example:
# nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
# return 5, nums[:5] == [0, 1, 2, 3, 4]


# ─── Sort Colors (Dutch National Flag) ───────────────────────────────────────
# 3-way partition: 0s to left, 2s to right, 1s in middle. One pass.

def sort_colors(nums: List[int]) -> None:
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1

# Example:
# nums = [2, 0, 2, 1, 1, 0] -> [0, 0, 1, 1, 2, 2]


# ─── Subarray Product Less Than K ─────────────────────────────────────────────
# Count subarrays where product of all elements < k. Sliding window with two ptrs.

def num_subarrays_product_less_than_k(nums: List[int], k: int) -> int:
    if k <= 1:
        return 0
    count = 0
    product = 1
    lo = 0

    for hi in range(len(nums)):
        product *= nums[hi]
        while product >= k:
            product //= nums[lo]
            lo += 1
        count += hi - lo + 1  # all subarrays ending at hi with start in [lo, hi]

    return count

# Example:
# nums = [10, 5, 2, 6], k = 100
# answer -> 8


# ─── Minimum Size Subarray Sum ────────────────────────────────────────────────
# Shortest subarray with sum >= target (all positive nums).

def min_size_subarray_sum(target: int, nums: List[int]) -> int:
    lo = 0
    curr_sum = 0
    min_len = float('inf')

    for hi in range(len(nums)):
        curr_sum += nums[hi]
        while curr_sum >= target:
            min_len = min(min_len, hi - lo + 1)
            curr_sum -= nums[lo]
            lo += 1

    return min_len if min_len != float('inf') else 0

# Example:
# target = 7, nums = [2, 3, 1, 2, 4, 3]
# answer -> 2 (subarray [4, 3])


if __name__ == "__main__":
    assert pair_sum_sorted([2, 7, 11, 15], 9) == (1, 2)
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert three_sum_closest([-1, 2, 1, -4], 1) == 2
    assert max_water_container([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert num_subarrays_product_less_than_k([10, 5, 2, 6], 100) == 8
    assert min_size_subarray_sum(7, [2, 3, 1, 2, 4, 3]) == 2
    print("All tests passed.")
