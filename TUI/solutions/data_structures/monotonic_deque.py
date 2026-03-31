"""
NAME: Monotonic Deque (Sliding Window Maximum / Minimum)
TAGS: deque, sliding window, array, monotonic
DESCRIPTION: A double-ended queue that maintains elements in monotonically
             decreasing (for max) or increasing (for min) order within a window.
             Use for O(n) sliding window extrema problems that would otherwise be O(nk).
COMPLEXITY: Time O(n), Space O(k) where k is the window size
"""

from collections import deque
from typing import List


# ─── Sliding Window Maximum ───────────────────────────────────────────────────
# Find the maximum in every contiguous subarray of size k.
# Deque stores indices; front always holds the index of the current window max.

def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
    result = []
    dq = deque()  # stores indices, decreasing order of values

    for i in range(len(nums)):
        # Remove indices outside the current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Maintain decreasing order: remove smaller elements from back
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Start recording once first window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Example:
# nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
# result -> [3, 3, 5, 5, 6, 7]


# ─── Sliding Window Minimum ───────────────────────────────────────────────────
# Find the minimum in every contiguous subarray of size k.

def sliding_window_minimum(nums: List[int], k: int) -> List[int]:
    result = []
    dq = deque()  # stores indices, increasing order of values

    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Maintain increasing order: remove larger elements from back
        while dq and nums[dq[-1]] > nums[i]:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Example:
# nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
# result -> [-1, -3, -3, -3, 3, 3]


# ─── Sliding Window Max and Min (combined) ────────────────────────────────────
def sliding_window_max_min(nums: List[int], k: int):
    """Returns (max_list, min_list) for all windows of size k."""
    n = len(nums)
    max_result, min_result = [], []
    max_dq, min_dq = deque(), deque()

    for i in range(n):
        # Evict out-of-window indices
        if max_dq and max_dq[0] < i - k + 1:
            max_dq.popleft()
        if min_dq and min_dq[0] < i - k + 1:
            min_dq.popleft()

        while max_dq and nums[max_dq[-1]] < nums[i]:
            max_dq.pop()
        while min_dq and nums[min_dq[-1]] > nums[i]:
            min_dq.pop()

        max_dq.append(i)
        min_dq.append(i)

        if i >= k - 1:
            max_result.append(nums[max_dq[0]])
            min_result.append(nums[min_dq[0]])

    return max_result, min_result


# ─── Constrained Subsequence Sum ──────────────────────────────────────────────
# Max sum of subsequence where no two chosen elements are more than k apart.
# dp[i] = nums[i] + max(dp[i-k..i-1], 0)
# Deque maintains max of dp over a window of size k.

def constrained_subsequence_sum(nums: List[int], k: int) -> int:
    n = len(nums)
    dp = [0] * n
    dq = deque()  # stores indices, decreasing order of dp values

    for i in range(n):
        # Evict indices outside window [i-k, i-1]
        while dq and dq[0] < i - k:
            dq.popleft()

        best_prev = dp[dq[0]] if dq else 0
        dp[i] = nums[i] + max(0, best_prev)

        # Maintain decreasing order
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)

    return max(dp)

# Example:
# nums = [10, 2, -10, 5, 20], k = 2
# answer -> 37  (10 + 2 + 5 + 20)


# ─── Jump Game VI (DP + Monotonic Deque) ─────────────────────────────────────
# You can jump up to k steps forward. Maximize score (sum of visited cells).
# dp[i] = nums[i] + max(dp[i-k..i-1])

def max_result(nums: List[int], k: int) -> int:
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dq = deque([0])

    for i in range(1, n):
        while dq and dq[0] < i - k:
            dq.popleft()

        dp[i] = nums[i] + dp[dq[0]]

        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)

    return dp[-1]

# Example:
# nums = [1, -1, -2, 4, -7, 3], k = 2
# answer -> 7  (path: 1 -> 4 -> 3)


# ─── Shortest Subarray with Sum >= K (including negatives) ────────────────────
# Classic problem: prefix sums + monotonic deque for O(n) solution.
# Works even with negative numbers (unlike standard sliding window).

def shortest_subarray_sum_k(nums: List[int], k: int) -> int:
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    result = float('inf')
    dq = deque()  # stores indices of prefix, increasing order of prefix values

    for i in range(n + 1):
        # prefix[i] - prefix[dq[0]] >= k means subarray [dq[0]..i-1] has sum >= k
        while dq and prefix[i] - prefix[dq[0]] >= k:
            result = min(result, i - dq.popleft())

        # Maintain increasing prefix sums in deque
        while dq and prefix[dq[-1]] >= prefix[i]:
            dq.pop()
        dq.append(i)

    return result if result != float('inf') else -1

# Example:
# nums = [2, -1, 2], k = 3
# answer -> 3


# ─── Monotonic Deque as a Generic Template ────────────────────────────────────
class MonotonicDeque:
    """
    Generic monotonic deque template.
    mode='max' -> front always holds index of current window maximum
    mode='min' -> front always holds index of current window minimum
    """
    def __init__(self, mode: str = 'max'):
        self.dq = deque()
        self.cmp = (lambda a, b: a < b) if mode == 'max' else (lambda a, b: a > b)

    def push(self, i: int, val, window_start: int):
        # Evict out-of-window indices
        while self.dq and self.dq[0] < window_start:
            self.dq.popleft()
        # Maintain monotonic property
        while self.dq and self.cmp(self._vals[self.dq[-1]], val):
            self.dq.pop()
        self.dq.append(i)

    def peek_front_index(self) -> int:
        return self.dq[0]


if __name__ == "__main__":
    assert sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert sliding_window_minimum([1, 3, -1, -3, 5, 3, 6, 7], 3) == [-1, -3, -3, -3, 3, 3]
    assert constrained_subsequence_sum([10, 2, -10, 5, 20], 2) == 37
    assert max_result([1, -1, -2, 4, -7, 3], 2) == 7
    assert shortest_subarray_sum_k([2, -1, 2], 3) == 3
    print("All tests passed.")
