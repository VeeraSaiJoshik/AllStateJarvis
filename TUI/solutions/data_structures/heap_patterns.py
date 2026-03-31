"""
NAME: Heap Patterns
TAGS: heap, priority queue, greedy, sorting
DESCRIPTION: Common heap (priority queue) patterns: k-th largest/smallest,
             merging sorted sequences, top-k elements, and running median via two heaps.
             Use when you need repeated access to the min or max of a dynamic set in O(log n).
COMPLEXITY: Time O(n log k) typical, Space O(k) or O(n)
"""

import heapq
from typing import List, Iterator, Optional


# ─── K-th Largest Element ─────────────────────────────────────────────────────
# Maintain a min-heap of size k. The top is the k-th largest.

def kth_largest(nums: List[int], k: int) -> int:
    heap = []  # min-heap of size k
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]

# Example:
# nums = [3, 2, 1, 5, 6, 4], k = 2 -> 5


# ─── K-th Smallest Element ────────────────────────────────────────────────────
# Maintain a max-heap of size k (negate values for max-heap in Python).

def kth_smallest(nums: List[int], k: int) -> int:
    heap = []  # max-heap (negated) of size k
    for num in nums:
        heapq.heappush(heap, -num)
        if len(heap) > k:
            heapq.heappop(heap)
    return -heap[0]

# Example:
# nums = [3, 2, 1, 5, 6, 4], k = 2 -> 2


# ─── Top-K Frequent Elements ──────────────────────────────────────────────────

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    from collections import Counter
    freq = Counter(nums)
    # min-heap on frequency, keep top k
    heap = []
    for num, cnt in freq.items():
        heapq.heappush(heap, (cnt, num))
        if len(heap) > k:
            heapq.heappop(heap)
    return [num for cnt, num in heap]

# Example:
# nums = [1, 1, 1, 2, 2, 3], k = 2 -> [1, 2] (order may vary)


# ─── K Closest Points to Origin ───────────────────────────────────────────────

def k_closest_points(points: List[List[int]], k: int) -> List[List[int]]:
    # max-heap of size k (negate distance)
    heap = []
    for x, y in points:
        dist = x * x + y * y
        heapq.heappush(heap, (-dist, x, y))
        if len(heap) > k:
            heapq.heappop(heap)
    return [[x, y] for _, x, y in heap]

# Example:
# points = [[1,3],[-2,2]], k = 1 -> [[-2, 2]]


# ─── Merge K Sorted Lists ─────────────────────────────────────────────────────
# Use a min-heap storing (value, list_index, element_index).

def merge_k_sorted_lists(lists: List[List[int]]) -> List[int]:
    result = []
    heap = []

    # Initialize heap with first element of each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, i, j = heapq.heappop(heap)
        result.append(val)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j + 1], i, j + 1))

    return result

# Example:
# lists = [[1,4,5],[1,3,4],[2,6]] -> [1,1,2,3,4,4,5,6]


# ─── Merge K Sorted Iterators (generator version) ────────────────────────────

def merge_k_sorted_iterators(*iterables) -> Iterator:
    """Lazily merge k sorted iterables."""
    heap = []
    iterators = [iter(it) for it in iterables]
    for i, it in enumerate(iterators):
        val = next(it, None)
        if val is not None:
            heapq.heappush(heap, (val, i))

    while heap:
        val, i = heapq.heappop(heap)
        yield val
        nxt = next(iterators[i], None)
        if nxt is not None:
            heapq.heappush(heap, (nxt, i))


# ─── Running Median (Two Heaps) ───────────────────────────────────────────────
# Keep a max-heap for the lower half and a min-heap for the upper half.
# Invariant: len(lower) == len(upper) or len(lower) == len(upper) + 1.
# Median is always lower[0] (negated) or average of both tops.

class RunningMedian:
    def __init__(self):
        self.lower = []  # max-heap (negated), stores the smaller half
        self.upper = []  # min-heap, stores the larger half

    def add(self, num: int) -> None:
        # Always push to lower first
        heapq.heappush(self.lower, -num)

        # Balance: lower's max must be <= upper's min
        if self.upper and (-self.lower[0]) > self.upper[0]:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))

        # Balance sizes: lower can have at most 1 more element
        if len(self.lower) > len(self.upper) + 1:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        elif len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    def get_median(self) -> float:
        if len(self.lower) == len(self.upper):
            return (-self.lower[0] + self.upper[0]) / 2.0
        return float(-self.lower[0])

# Example:
# rm = RunningMedian()
# rm.add(1) -> median 1.0
# rm.add(2) -> median 1.5
# rm.add(3) -> median 2.0


# ─── Sliding Window Median ────────────────────────────────────────────────────
# Harder variant: maintain running median in a sliding window.
# Uses two heaps with lazy deletion.

def sliding_window_median(nums: List[int], k: int) -> List[float]:
    from collections import defaultdict

    lower = []   # max-heap (negated)
    upper = []   # min-heap
    lo_size = up_size = 0
    invalid = defaultdict(int)  # lazy deletion counts
    result = []

    def balance():
        nonlocal lo_size, up_size
        while lo_size > up_size + 1:
            heapq.heappush(upper, -heapq.heappop(lower))
            lo_size -= 1
            up_size += 1
            # clean top of lower
            while lower and invalid[-lower[0]] > 0:
                invalid[-heapq.heappop(lower)] -= 1
        while up_size > lo_size:
            heapq.heappush(lower, -heapq.heappop(upper))
            lo_size += 1
            up_size -= 1
            # clean top of upper
            while upper and invalid[upper[0]] > 0:
                invalid[heapq.heappop(upper)] -= 1

    for i, num in enumerate(nums):
        # Add new element
        heapq.heappush(lower, -num)
        lo_size += 1
        # Maintain order
        if upper and (-lower[0]) > upper[0]:
            heapq.heappush(upper, -heapq.heappop(lower))
            lo_size -= 1
            up_size += 1
        balance()

        if i >= k:
            # Remove outgoing element
            out = nums[i - k]
            invalid[out] += 1
            if out <= -lower[0]:
                lo_size -= 1
            else:
                up_size -= 1
            # Clean stale tops
            while lower and invalid[-lower[0]] > 0:
                invalid[-heapq.heappop(lower)] -= 1
            while upper and invalid[upper[0]] > 0:
                invalid[heapq.heappop(upper)] -= 1
            balance()

        if i >= k - 1:
            if k % 2 == 1:
                result.append(float(-lower[0]))
            else:
                result.append((-lower[0] + upper[0]) / 2.0)

    return result

# Example:
# nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
# result -> [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]


# ─── Task Scheduler ───────────────────────────────────────────────────────────
# Minimum intervals to finish all tasks with cooldown n between same tasks.
# Greedy: always pick the most frequent available task.

def least_interval(tasks: List[str], n: int) -> int:
    from collections import Counter
    freq = Counter(tasks)
    heap = [-cnt for cnt in freq.values()]
    heapq.heapify(heap)
    time = 0

    while heap:
        temp = []
        cycle = n + 1  # one cycle of n+1 slots
        for _ in range(cycle):
            if heap:
                cnt = heapq.heappop(heap)
                if cnt + 1 < 0:  # still has remaining tasks
                    temp.append(cnt + 1)
            time += 1
            if not heap and not temp:
                break
        for t in temp:
            heapq.heappush(heap, t)

    return time

# Example:
# tasks = ["A","A","A","B","B","B"], n = 2 -> 8


# ─── Find K Pairs with Smallest Sums ──────────────────────────────────────────
# Two sorted arrays, find k pairs (u, v) with smallest u + v.

def k_smallest_pairs(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    if not nums1 or not nums2:
        return []

    result = []
    # heap: (sum, i, j)
    heap = [(nums1[0] + nums2[j], 0, j) for j in range(min(k, len(nums2)))]
    heapq.heapify(heap)

    while heap and len(result) < k:
        s, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        if i + 1 < len(nums1):
            heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))

    return result

# Example:
# nums1 = [1, 7, 11], nums2 = [2, 4, 6], k = 3
# result -> [[1,2],[1,4],[1,6]]


# ─── Reorganize String ────────────────────────────────────────────────────────
# Arrange characters so no two adjacent are the same (greedy max-heap).

def reorganize_string(s: str) -> str:
    from collections import Counter
    freq = Counter(s)
    heap = [(-cnt, ch) for ch, cnt in freq.items()]
    heapq.heapify(heap)

    result = []
    prev_cnt, prev_ch = 0, ''

    while heap:
        cnt, ch = heapq.heappop(heap)
        result.append(ch)
        if prev_cnt < 0:
            heapq.heappush(heap, (prev_cnt, prev_ch))
        prev_cnt, prev_ch = cnt + 1, ch  # decrement (cnt is negative)

    res = ''.join(result)
    return res if len(res) == len(s) else ""

# Example:
# s = "aab" -> "aba"
# s = "aaab" -> ""


if __name__ == "__main__":
    assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
    assert kth_smallest([3, 2, 1, 5, 6, 4], 2) == 2
    assert sorted(top_k_frequent([1,1,1,2,2,3], 2)) == [1, 2]
    assert merge_k_sorted_lists([[1,4,5],[1,3,4],[2,6]]) == [1,1,2,3,4,4,5,6]

    rm = RunningMedian()
    rm.add(1); rm.add(2); rm.add(3)
    assert rm.get_median() == 2.0

    assert sliding_window_median([1,3,-1,-3,5,3,6,7], 3) == [1.0,-1.0,-1.0,3.0,5.0,6.0]
    assert least_interval(["A","A","A","B","B","B"], 2) == 8
    assert k_smallest_pairs([1,7,11],[2,4,6],3) == [[1,2],[1,4],[1,6]]
    print("All tests passed.")
