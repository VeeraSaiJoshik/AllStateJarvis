"""
NAME: Sparse Table
TAGS: sparse table, range minimum query, range maximum query, static array, idempotent
DESCRIPTION: Sparse table answers range minimum/maximum queries in O(1) after O(n log n)
    preprocessing on an immutable array. Use when the array never changes and you need
    millions of RMQ queries — it's faster than a segment tree for static data.
COMPLEXITY: Time O(n log n) build, O(1) query, Space O(n log n)
"""

import math
from typing import Callable, List


# ─────────────────────────────────────────────
# Generic Sparse Table (idempotent ops only)
# ─────────────────────────────────────────────
class SparseTable:
    """
    Works with any idempotent operation: min, max, gcd, bitwise AND/OR.
    Does NOT work correctly for sum (use a Fenwick/SegTree for that).
    """

    def __init__(self, data: List[int], op: Callable = min):
        self.n = n = len(data)
        self.op = op
        self.LOG = max(1, n.bit_length())           # floor(log2(n)) + 1
        # table[k][i] = op of data[i : i + 2^k]
        self.table = [[0] * n for _ in range(self.LOG)]
        self.table[0] = data[:]
        for k in range(1, self.LOG):
            half = 1 << (k - 1)
            for i in range(n - (1 << k) + 1):
                self.table[k][i] = op(self.table[k - 1][i],
                                      self.table[k - 1][i + half])
        # Precompute floor(log2(i)) for i in [1, n]
        self._log = [0] * (n + 1)
        for i in range(2, n + 1):
            self._log[i] = self._log[i >> 1] + 1

    def query(self, l: int, r: int) -> int:
        """
        O(1) query over [l, r] inclusive.
        For idempotent ops the two overlapping halves give the correct answer.
        """
        k = self._log[r - l + 1]
        return self.op(self.table[k][l], self.table[k][r - (1 << k) + 1])


# ─────────────────────────────────────────────
# RMQ that also returns the index of the extremum
# ─────────────────────────────────────────────
class SparseTableIdx:
    """
    Returns (value, leftmost_index) for range minimum queries.
    Ties broken by taking the smaller index.
    """

    def __init__(self, data: List[int]):
        self.n = n = len(data)
        self.LOG = max(1, n.bit_length())
        self.table = [[(0, 0)] * n for _ in range(self.LOG)]
        self.table[0] = [(v, i) for i, v in enumerate(data)]
        for k in range(1, self.LOG):
            half = 1 << (k - 1)
            for i in range(n - (1 << k) + 1):
                self.table[k][i] = min(self.table[k - 1][i],
                                       self.table[k - 1][i + half])
        self._log = [0] * (n + 1)
        for i in range(2, n + 1):
            self._log[i] = self._log[i >> 1] + 1

    def query(self, l: int, r: int):
        """Returns (min_value, leftmost_index) over [l, r]."""
        k = self._log[r - l + 1]
        return min(self.table[k][l], self.table[k][r - (1 << k) + 1])


# ─────────────────────────────────────────────
# Disjoint Sparse Table — supports non-idempotent
# ops (e.g. sum, product) with O(1) query
# ─────────────────────────────────────────────
class DisjointSparseTable:
    """
    Supports ANY associative operation (including sum, product).
    O(n log n) build, O(1) query.
    Reference: https://codeforces.com/blog/entry/79108
    """

    def __init__(self, data: List[int], op: Callable = lambda a, b: a + b, identity: int = 0):
        self.n = n = len(data)
        self.op = op
        self.e = identity
        self.LOG = max(1, (n - 1).bit_length() + 1) if n > 1 else 1
        self.table = [[identity] * n for _ in range(self.LOG)]
        self.table[0] = data[:]
        for k in range(1, self.LOG):
            half = 1 << k
            for mid in range(half, n, half << 1):
                # Build left half (mid-1 down to mid-half)
                self.table[k][mid - 1] = data[mid - 1]
                for i in range(mid - 2, mid - half - 1, -1):
                    if i < 0:
                        break
                    self.table[k][i] = op(data[i], self.table[k][i + 1])
                # Build right half (mid up to mid+half-1)
                self.table[k][mid] = data[mid] if mid < n else identity
                for i in range(mid + 1, min(mid + half, n)):
                    self.table[k][i] = op(self.table[k][i - 1], data[i])

    def query(self, l: int, r: int) -> int:
        """Query op over [l, r] inclusive."""
        if l == r:
            return self.table[0][l]
        k = (l ^ r).bit_length() - 1
        # Careful: for k==0 the above formula gives 0 but l!=r handled
        k = max(k, 1)
        return self.op(self.table[k][l], self.table[k][r])


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    arr = [2, 4, 3, 1, 6, 7, 8, 9, 1, 7]

    # Range min
    st_min = SparseTable(arr, min)
    assert st_min.query(0, 9) == 1
    assert st_min.query(1, 4) == 1
    assert st_min.query(5, 8) == 1

    # Range max
    st_max = SparseTable(arr, max)
    assert st_max.query(0, 9) == 9
    assert st_max.query(0, 3) == 4

    # Range min with index
    sti = SparseTableIdx(arr)
    assert sti.query(0, 3) == (1, 3)     # min=1 at index 3
    assert sti.query(5, 7) == (7, 5)     # min=7 at index 5

    # Disjoint sparse table (sum)
    dst = DisjointSparseTable(arr, lambda a, b: a + b, 0)
    assert dst.query(0, 4) == 16         # 2+4+3+1+6
    assert dst.query(3, 3) == 1

    print("All sparse table tests passed.")
