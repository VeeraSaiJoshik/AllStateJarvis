"""
NAME: Segment Tree with Lazy Propagation
TAGS: segment tree, lazy propagation, range update, range query, data structures
DESCRIPTION: Segment tree that supports range updates (add to all elements in a range)
    and range queries (sum/min/max) in O(log n) each. Use when both range updates and
    range queries are needed simultaneously — lazy propagation defers work until necessary.
COMPLEXITY: Time O(n) build, O(log n) range update/query, Space O(n)
"""

from typing import List


# ─────────────────────────────────────────────
# Lazy Seg Tree — Range Add + Range Sum
# ─────────────────────────────────────────────
class LazySegTree:
    """
    Supports:
      range_add(l, r, val)  — add val to every element in [l, r]
      range_sum(l, r)        — sum of elements in [l, r]
    """

    def __init__(self, data: List[int]):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data: List[int], node: int, start: int, end: int) -> None:
        if start == end:
            self.tree[node] = data[start]
        else:
            mid = (start + end) // 2
            self._build(data, 2 * node, start, mid)
            self._build(data, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node: int, start: int, end: int) -> None:
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            lc, rc = 2 * node, 2 * node + 1
            self.tree[lc] += self.lazy[node] * (mid - start + 1)
            self.tree[rc] += self.lazy[node] * (end - mid)
            self.lazy[lc] += self.lazy[node]
            self.lazy[rc] += self.lazy[node]
            self.lazy[node] = 0

    def _update(self, node: int, start: int, end: int, l: int, r: int, val: int) -> None:
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._update(2 * node, start, mid, l, r, val)
        self._update(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, l, r) +
                self._query(2 * node + 1, mid + 1, end, l, r))

    def range_add(self, l: int, r: int, val: int) -> None:
        """Add val to all elements in [l, r]."""
        self._update(1, 0, self.n - 1, l, r, val)

    def range_sum(self, l: int, r: int) -> int:
        """Sum of elements in [l, r]."""
        return self._query(1, 0, self.n - 1, l, r)

    def point_update(self, i: int, val: int) -> None:
        """Set arr[i] = val."""
        # Get current value then adjust
        cur = self.range_sum(i, i)
        self.range_add(i, i, val - cur)


# ─────────────────────────────────────────────
# Lazy Seg Tree — Range Assign + Range Min/Max
# ─────────────────────────────────────────────
class LazyAssignMinTree:
    """
    Supports:
      range_assign(l, r, val) — set all elements in [l, r] to val
      range_min(l, r)          — minimum of elements in [l, r]
    Lazy value = None means no pending assignment.
    """

    INF = float("inf")

    def __init__(self, data: List[int]):
        self.n = len(data)
        self.tree = [self.INF] * (4 * self.n)
        self.lazy = [None] * (4 * self.n)   # None = no pending assign
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data: List[int], node: int, start: int, end: int) -> None:
        if start == end:
            self.tree[node] = data[start]
        else:
            mid = (start + end) // 2
            self._build(data, 2 * node, start, mid)
            self._build(data, 2 * node + 1, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _push_down(self, node: int) -> None:
        if self.lazy[node] is not None:
            for child in (2 * node, 2 * node + 1):
                self.tree[child] = self.lazy[node]
                self.lazy[child] = self.lazy[node]
            self.lazy[node] = None

    def _update(self, node: int, start: int, end: int, l: int, r: int, val: int) -> None:
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] = val
            self.lazy[node] = val
            return
        self._push_down(node)
        mid = (start + end) // 2
        self._update(2 * node, start, mid, l, r, val)
        self._update(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        if r < start or end < l:
            return self.INF
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node)
        mid = (start + end) // 2
        return min(self._query(2 * node, start, mid, l, r),
                   self._query(2 * node + 1, mid + 1, end, l, r))

    def range_assign(self, l: int, r: int, val: int) -> None:
        """Assign val to all elements in [l, r]."""
        self._update(1, 0, self.n - 1, l, r, val)

    def range_min(self, l: int, r: int) -> int:
        """Minimum of elements in [l, r]."""
        return self._query(1, 0, self.n - 1, l, r)


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    arr = [1, 2, 3, 4, 5]

    # Range add + range sum
    lst = LazySegTree(arr)
    assert lst.range_sum(0, 4) == 15
    lst.range_add(1, 3, 10)              # arr becomes [1,12,13,14,5]
    assert lst.range_sum(1, 3) == 39
    assert lst.range_sum(0, 4) == 45

    # Range assign + range min
    lmt = LazyAssignMinTree(arr)
    assert lmt.range_min(0, 4) == 1
    lmt.range_assign(2, 4, 1)           # arr becomes [1,2,1,1,1]
    assert lmt.range_min(0, 4) == 1
    assert lmt.range_min(1, 2) == 1

    print("All lazy segment tree tests passed.")
