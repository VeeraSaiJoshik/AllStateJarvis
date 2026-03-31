"""
NAME: Coordinate Compression and Sweep Line
TAGS: coordinate-compression, sweep-line, offline, intervals
DESCRIPTION: Compress large coordinate values to small indices for array-based data structures.
             Sweep line with event sorting solves interval overlap, area, and intersection problems.
             Essential when coordinates are up to 10^9 but count is small (< 10^6).
COMPLEXITY: Compression O(n log n); Sweep line O(n log n)
"""

from typing import List, Tuple, Dict
from bisect import bisect_left, insort


# ─── Coordinate Compression ───────────────────────────────────────────────────

def compress(values: List[int]) -> Tuple[List[int], Dict[int, int]]:
    """
    Returns (sorted_unique, rank_map) where rank_map[v] = 0-indexed compressed rank.
    """
    sorted_unique = sorted(set(values))
    rank_map = {v: i for i, v in enumerate(sorted_unique)}
    return sorted_unique, rank_map

def compress_array(arr: List[int]) -> List[int]:
    """Replace each value with its 0-indexed rank."""
    _, rank_map = compress(arr)
    return [rank_map[v] for v in arr]

# Example:
# arr = [40, 10, 20, 10, 30]
# compress_array(arr) -> [3, 0, 1, 0, 2]


def compress_with_query(values: List[int], queries: List[int]) -> Tuple[List[int], List[int]]:
    """Compress both values and queries on the same coordinate space."""
    all_vals = sorted(set(values) | set(queries))
    rank_map = {v: i for i, v in enumerate(all_vals)}
    return [rank_map[v] for v in values], [rank_map[v] for v in queries]


# ─── Fenwick Tree (BIT) for use after compression ─────────────────────────────

class BIT:
    """Binary Indexed Tree (Fenwick Tree) — 1-indexed."""
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int = 1) -> None:
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        """Prefix sum [1..i]."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_query(self, lo: int, hi: int) -> int:
        return self.query(hi) - self.query(lo - 1)


# ─── Count Inversions via Compression + BIT ───────────────────────────────────

def count_inversions_bit(arr: List[int]) -> int:
    """O(n log n) inversion count using coordinate compression + BIT."""
    compressed = compress_array(arr)
    n = len(arr)
    bit = BIT(n)
    inversions = 0

    for val in compressed:
        # Count elements already inserted that are greater than val
        inversions += bit.range_query(val + 2, n)  # +2 because 1-indexed and val is 0-indexed
        bit.update(val + 1)  # +1 for 1-indexed BIT

    return inversions

# Example:
# arr = [2, 4, 1, 3, 5] -> 3


# ─── Offline Sweep Line: Merge Intervals ──────────────────────────────────────

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

# Example:
# intervals = [[1,3],[2,6],[8,10],[15,18]] -> [[1,6],[8,10],[15,18]]


# ─── Sweep Line: Maximum Simultaneous Intervals ───────────────────────────────
# Events: +1 at start, -1 at end+1. Sort events and sweep.

def max_simultaneous_intervals(intervals: List[List[int]]) -> int:
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end + 1, -1))  # exclusive end

    events.sort()
    max_overlap = curr = 0
    for _, delta in events:
        curr += delta
        max_overlap = max(max_overlap, curr)
    return max_overlap

# Example:
# intervals = [[1,4],[2,5],[9,12],[5,9],[5,12]] -> 3


# ─── Sweep Line: Count Interval Overlaps at Each Point ────────────────────────

def intervals_coverage(intervals: List[List[int]], points: List[int]) -> List[int]:
    """For each point, count how many intervals contain it."""
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end + 1, -1))

    events.sort()
    points_sorted = sorted((p, i) for i, p in enumerate(points))
    result = [0] * len(points)
    curr = 0
    ei = 0

    for p, idx in points_sorted:
        while ei < len(events) and events[ei][0] <= p:
            curr += events[ei][1]
            ei += 1
        result[idx] = curr

    return result


# ─── Sweep Line: Rectangle Union Area ─────────────────────────────────────────
# Given axis-aligned rectangles, compute total area covered (union).
# Uses coordinate compression on y-axis + sweep on x-axis.

def rectangle_union_area(rects: List[List[int]]) -> int:
    """
    Each rect: [x1, y1, x2, y2] (bottom-left to top-right).
    Returns the area of their union.
    """
    # Collect all y-coordinates and compress
    ys = sorted(set(y for x1, y1, x2, y2 in rects for y in (y1, y2)))
    y_rank = {y: i for i, y in enumerate(ys)}
    num_y = len(ys)

    # Build events: (x, type, y1, y2)
    # type +1 = enter, -1 = exit
    events = []
    for x1, y1, x2, y2 in rects:
        events.append((x1,  1, y_rank[y1], y_rank[y2]))
        events.append((x2, -1, y_rank[y1], y_rank[y2]))
    events.sort()

    # Segment tree / count array to track active y-segments
    count = [0] * (num_y - 1)  # each slot covers ys[i]..ys[i+1]

    def covered_length() -> int:
        total = 0
        for i in range(num_y - 1):
            if count[i] > 0:
                total += ys[i + 1] - ys[i]
        return total

    total_area = 0
    prev_x = events[0][0]

    for x, typ, y1, y2 in events:
        total_area += covered_length() * (x - prev_x)
        for i in range(y1, y2):
            count[i] += typ
        prev_x = x

    return total_area

# Example:
# rects = [[0,0,2,2],[1,1,3,3]] -> 7


# ─── Coordinate Compression for 2D Grid ──────────────────────────────────────

def compress_2d(points: List[Tuple[int, int]]):
    """Compress x and y coordinates independently."""
    xs = sorted(set(p[0] for p in points))
    ys = sorted(set(p[1] for p in points))
    x_rank = {v: i for i, v in enumerate(xs)}
    y_rank = {v: i for i, v in enumerate(ys)}
    compressed = [(x_rank[x], y_rank[y]) for x, y in points]
    return compressed, x_rank, y_rank


# ─── Offline Interval Query: Number of Distinct in Range ─────────────────────
# For queries [l, r], count distinct values in arr[l..r].
# Sort queries by right endpoint, maintain last occurrence.

def count_distinct_range(arr: List[int], queries: List[Tuple[int, int]]) -> List[int]:
    """
    For each query (l, r), return the count of distinct values in arr[l..r].
    Offline: sort by r, use BIT to count active elements.
    """
    n = len(arr)
    bit = BIT(n)
    last_seen = {}
    results = [0] * len(queries)

    # Sort queries by right endpoint
    sorted_queries = sorted(enumerate(queries), key=lambda x: x[1][1])
    arr_idx = 0

    for qi, (l, r) in sorted_queries:
        # Process all elements up to index r
        while arr_idx <= r:
            v = arr[arr_idx]
            if v in last_seen:
                bit.update(last_seen[v] + 1, -1)  # remove old occurrence
            bit.update(arr_idx + 1, 1)             # add new occurrence
            last_seen[v] = arr_idx
            arr_idx += 1
        results[qi] = bit.range_query(l + 1, r + 1)

    return results

# Example:
# arr = [1, 2, 1, 3, 2]
# queries = [(0, 4), (1, 3), (2, 4)]
# results -> [3, 3, 3]  (all windows contain 3 distinct values)


# ─── Meeting Rooms II (min rooms needed) ──────────────────────────────────────
# Same as max simultaneous intervals.

def min_meeting_rooms(intervals: List[List[int]]) -> int:
    return max_simultaneous_intervals(intervals)

# Example:
# intervals = [[0,30],[5,10],[15,20]] -> 2


# ─── Insert Interval ──────────────────────────────────────────────────────────
def insert_interval(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    result = []
    i = 0
    n = len(intervals)

    # Add all intervals that end before new_interval starts
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)

    # Add remaining intervals
    result.extend(intervals[i:])
    return result

# Example:
# intervals = [[1,3],[6,9]], new_interval = [2,5] -> [[1,5],[6,9]]


if __name__ == "__main__":
    assert compress_array([40, 10, 20, 10, 30]) == [3, 0, 1, 0, 2]
    assert count_inversions_bit([2, 4, 1, 3, 5]) == 3
    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    assert max_simultaneous_intervals([[1,4],[2,5],[9,12],[5,9],[5,12]]) == 3
    assert rectangle_union_area([[0,0,2,2],[1,1,3,3]]) == 7
    assert count_distinct_range([1,2,1,3,2], [(0,4),(1,3),(2,4)]) == [3, 3, 3]
    assert min_meeting_rooms([[0,30],[5,10],[15,20]]) == 2
    assert insert_interval([[1,3],[6,9]], [2,5]) == [[1,5],[6,9]]
    print("All tests passed.")
