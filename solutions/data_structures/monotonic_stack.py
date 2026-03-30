"""
NAME: Monotonic Stack
TAGS: stack, array, greedy, histogram
DESCRIPTION: A stack that maintains elements in monotonically increasing or decreasing order.
             Use for next greater/smaller element problems, histogram area, and span queries.
             Essential in competitions for O(n) solutions to problems that seem to require O(n^2).
COMPLEXITY: Time O(n), Space O(n)
"""

from typing import List


# ─── Next Greater Element ───────────────────────────────────────────────────
# For each element, find the next element to its right that is strictly greater.
# Returns -1 if no such element exists.

def next_greater_element(arr: List[int]) -> List[int]:
    n = len(arr)
    result = [-1] * n
    stack = []  # stores indices, stack bottom has greatest values

    for i in range(n):
        # while stack is not empty and current element > element at stack top
        while stack and arr[i] > arr[stack[-1]]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result

# Example:
# arr = [4, 5, 2, 10, 8]
# result -> [5, 10, 10, -1, -1]


# ─── Next Smaller Element ────────────────────────────────────────────────────
# For each element, find the next element to its right that is strictly smaller.

def next_smaller_element(arr: List[int]) -> List[int]:
    n = len(arr)
    result = [-1] * n
    stack = []  # stores indices, stack bottom has smallest values

    for i in range(n):
        while stack and arr[i] < arr[stack[-1]]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result

# Example:
# arr = [4, 5, 2, 10, 8]
# result -> [2, 2, -1, 8, -1]


# ─── Previous Greater Element ────────────────────────────────────────────────
def previous_greater_element(arr: List[int]) -> List[int]:
    n = len(arr)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and arr[stack[-1]] <= arr[i]:
            stack.pop()
        if stack:
            result[i] = arr[stack[-1]]
        stack.append(i)

    return result


# ─── Previous Smaller Element ────────────────────────────────────────────────
def previous_smaller_element(arr: List[int]) -> List[int]:
    n = len(arr)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        if stack:
            result[i] = arr[stack[-1]]
        stack.append(i)

    return result


# ─── Largest Rectangle in Histogram ──────────────────────────────────────────
# Given heights of histogram bars, find the largest rectangle area.
# Key idea: for each bar, find left and right boundaries using monotonic stack.

def largest_rectangle_histogram(heights: List[int]) -> int:
    stack = []  # monotonically increasing stack of indices
    max_area = 0
    n = len(heights)

    for i in range(n + 1):
        h = 0 if i == n else heights[i]
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area

# Example:
# heights = [2, 1, 5, 6, 2, 3]
# answer  -> 10  (bars 5 and 6, width 2)


# ─── Maximal Rectangle in Binary Matrix ──────────────────────────────────────
# Reduce to largest rectangle in histogram row by row.

def maximal_rectangle(matrix: List[List[int]]) -> int:
    if not matrix or not matrix[0]:
        return 0

    n_cols = len(matrix[0])
    heights = [0] * n_cols
    max_area = 0

    for row in matrix:
        for j in range(n_cols):
            heights[j] = heights[j] + 1 if row[j] == 1 else 0
        max_area = max(max_area, largest_rectangle_histogram(heights))

    return max_area


# ─── Stock Span Problem ───────────────────────────────────────────────────────
# For each day, find the span: number of consecutive days (including today)
# where price was <= today's price.

def stock_span(prices: List[int]) -> List[int]:
    n = len(prices)
    span = [1] * n
    stack = []  # stores indices where prices are in decreasing order

    for i in range(n):
        while stack and prices[stack[-1]] <= prices[i]:
            stack.pop()
        span[i] = i + 1 if not stack else i - stack[-1]
        stack.append(i)

    return span

# Example:
# prices = [100, 80, 60, 70, 60, 75, 85]
# span   -> [1,   1,  1,  2,  1,  4,  6]


# ─── Sum of Subarray Minimums ─────────────────────────────────────────────────
# Sum of min(subarray) for all contiguous subarrays. Use contribution technique.
# For each element, find how many subarrays it is the minimum of.

def sum_subarray_minimums(arr: List[int]) -> int:
    MOD = 10**9 + 7
    n = len(arr)

    # left[i] = distance to previous smaller or equal element (default: i+1)
    # right[i] = distance to next smaller element (default: n-i)
    left = [0] * n
    right = [0] * n
    stack = []

    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        left[i] = i + 1 if not stack else i - stack[-1]
        stack.append(i)

    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        right[i] = n - i if not stack else stack[-1] - i
        stack.append(i)

    return sum(arr[i] * left[i] * right[i] for i in range(n)) % MOD

# Example:
# arr    = [3, 1, 2, 4]
# answer -> 17  (sum of all subarray mins)


# ─── Remove K Digits to Make Smallest Number ─────────────────────────────────
# Greedy with monotonic stack: keep digits in increasing order.

def remove_k_digits(num: str, k: int) -> str:
    stack = []
    for digit in num:
        while k and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    # if k still > 0, remove from end
    stack = stack[:-k] if k else stack
    return ''.join(stack).lstrip('0') or '0'

# Example:
# num = "1432219", k = 3  -> "1219"
# num = "10200",   k = 1  -> "200" -> "0" wait -> "200"


# ─── Daily Temperatures ───────────────────────────────────────────────────────
# How many days until a warmer temperature? Classic next-greater variant.

def daily_temperatures(temperatures: List[int]) -> List[int]:
    n = len(temperatures)
    result = [0] * n
    stack = []

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)

    return result

# Example:
# temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
# result       -> [1,  1,  4,  2,  1,  1,  0,  0]


if __name__ == "__main__":
    # Quick smoke tests
    assert next_greater_element([4, 5, 2, 10, 8]) == [5, 10, 10, -1, -1]
    assert next_smaller_element([4, 5, 2, 10, 8]) == [2, 2, -1, 8, -1]
    assert largest_rectangle_histogram([2, 1, 5, 6, 2, 3]) == 10
    assert stock_span([100, 80, 60, 70, 60, 75, 85]) == [1, 1, 1, 2, 1, 4, 6]
    assert sum_subarray_minimums([3, 1, 2, 4]) == 17
    assert remove_k_digits("1432219", 3) == "1219"
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    print("All tests passed.")
