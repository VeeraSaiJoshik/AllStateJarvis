"""
NAME: Fenwick Tree (Binary Indexed Tree / BIT)
TAGS: fenwick tree, BIT, prefix sum, range sum, 2D BIT, data structures
DESCRIPTION: Fenwick Tree supports prefix sum queries and point updates in O(log n)
    with extremely small constants and simple code — ideal for competitive programming.
    Includes a 2D variant for grid prefix sums and a range-update point-query variant.
COMPLEXITY: Time O(n) build, O(log n) update/query, Space O(n); 2D: O(nm) build, O(log n * log m)
"""

from typing import List


# ─────────────────────────────────────────────
# 1D Fenwick Tree — Point Update, Prefix Sum
# ─────────────────────────────────────────────
class BIT:
    """1-indexed internally. Pass 0-indexed data to __init__."""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    @classmethod
    def from_list(cls, data: List[int]) -> "BIT":
        bit = cls(len(data))
        for i, v in enumerate(data):
            bit.update(i, v)
        return bit

    def update(self, i: int, delta: int) -> None:
        """Add delta to index i (0-indexed)."""
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """Sum of [0, i] inclusive (0-indexed)."""
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def query(self, l: int, r: int) -> int:
        """Sum of [l, r] inclusive (0-indexed)."""
        return self.prefix(r) - (self.prefix(l - 1) if l > 0 else 0)

    def find_kth(self, k: int) -> int:
        """
        Find smallest index i s.t. prefix(i) >= k (1-indexed k).
        Requires all values >= 0. O(log n).
        """
        pos = 0
        log = self.n.bit_length()
        for i in range(log, -1, -1):
            nxt = pos + (1 << i)
            if nxt <= self.n and self.tree[nxt] < k:
                pos = nxt
                k -= self.tree[nxt]
        return pos  # 0-indexed


# ─────────────────────────────────────────────
# 1D Fenwick Tree — Range Update, Point Query
# Using the difference array trick on a BIT
# ─────────────────────────────────────────────
class BITRangeUpdate:
    """
    range_add(l, r, val)  — add val to [l, r]
    point_query(i)         — get value at i
    """

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 2)

    def _update(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def _prefix(self, i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_add(self, l: int, r: int, val: int) -> None:
        """Add val to all positions in [l, r] (0-indexed)."""
        self._update(l, val)
        if r + 1 <= self.n - 1:
            self._update(r + 1, -val)

    def point_query(self, i: int) -> int:
        """Get value at index i (0-indexed)."""
        return self._prefix(i)


# ─────────────────────────────────────────────
# 2D Fenwick Tree — Point Update, Rectangle Sum
# ─────────────────────────────────────────────
class BIT2D:
    """
    2D BIT for point updates and rectangle prefix sum queries.
    All indices are 0-based externally.
    """

    def __init__(self, rows: int, cols: int):
        self.R = rows
        self.C = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, r: int, c: int, delta: int) -> None:
        """Add delta at cell (r, c)."""
        r += 1
        while r <= self.R:
            cc = c + 1
            while cc <= self.C:
                self.tree[r][cc] += delta
                cc += cc & (-cc)
            r += r & (-r)

    def prefix(self, r: int, c: int) -> int:
        """Sum of rectangle (0,0) to (r,c) inclusive."""
        r += 1
        s = 0
        while r > 0:
            cc = c + 1
            while cc > 0:
                s += self.tree[r][cc]
                cc -= cc & (-cc)
            r -= r & (-r)
        return s

    def query(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Sum of sub-rectangle (r1,c1) to (r2,c2) inclusive."""
        res = self.prefix(r2, c2)
        if r1 > 0:
            res -= self.prefix(r1 - 1, c2)
        if c1 > 0:
            res -= self.prefix(r2, c1 - 1)
        if r1 > 0 and c1 > 0:
            res += self.prefix(r1 - 1, c1 - 1)
        return res


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1D BIT
    bit = BIT.from_list([1, 2, 3, 4, 5])
    assert bit.prefix(4) == 15
    assert bit.query(1, 3) == 9          # 2+3+4
    bit.update(2, 5)                      # arr[2] += 5 => 8
    assert bit.query(1, 3) == 14         # 2+8+4

    # find_kth: smallest index where prefix >= k
    bit2 = BIT.from_list([1, 1, 1, 1, 1])
    assert bit2.find_kth(3) == 2         # 0-indexed: positions 0,1,2 give prefix=3

    # Range update, point query
    bru = BITRangeUpdate(5)
    bru.range_add(1, 3, 10)
    assert bru.point_query(0) == 0
    assert bru.point_query(2) == 10
    assert bru.point_query(4) == 0

    # 2D BIT
    grid = BIT2D(3, 3)
    grid.update(0, 0, 1)
    grid.update(1, 1, 2)
    grid.update(2, 2, 3)
    assert grid.query(0, 0, 2, 2) == 6
    assert grid.query(1, 1, 2, 2) == 5

    print("All Fenwick tree tests passed.")
