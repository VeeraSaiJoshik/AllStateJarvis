"""
NAME: Segment Tree (Iterative)
TAGS: segment tree, range query, range sum, range min/max, data structures
DESCRIPTION: Iterative segment tree supporting range sum, range min, and range max queries
    with point updates in O(log n). Use when you need fast range queries on a mutable array.
    The iterative version is ~2x faster than recursive and preferred in competitions.
COMPLEXITY: Time O(n) build, O(log n) update/query, Space O(n)
"""

from typing import Callable, List, TypeVar

T = TypeVar("T")


# ─────────────────────────────────────────────
# Generic Iterative Segment Tree
# ─────────────────────────────────────────────
class SegTree:
    """
    Generic iterative segment tree.
    op    : combine function, e.g. (a,b)->a+b  or min or max
    identity : identity element for op, e.g. 0 for sum, inf for min
    """

    def __init__(self, data: List[int], op: Callable = lambda a, b: a + b, identity: int = 0):
        self.n = len(data)
        self.op = op
        self.e = identity
        self.tree = [identity] * (2 * self.n)
        # Build: fill leaves then internal nodes
        for i, v in enumerate(data):
            self.tree[self.n + i] = v
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = op(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, i: int, val: int) -> None:
        """Point update: set index i to val."""
        i += self.n
        self.tree[i] = val
        while i > 1:
            i >>= 1
            self.tree[i] = self.op(self.tree[2 * i], self.tree[2 * i + 1])

    def query(self, l: int, r: int) -> int:
        """Query op over [l, r] inclusive."""
        res = self.e
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                res = self.op(res, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = self.op(res, self.tree[r])
            l >>= 1
            r >>= 1
        return res


# ─────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────
class SumTree(SegTree):
    def __init__(self, data: List[int]):
        super().__init__(data, lambda a, b: a + b, 0)


class MinTree(SegTree):
    def __init__(self, data: List[int]):
        super().__init__(data, min, float("inf"))


class MaxTree(SegTree):
    def __init__(self, data: List[int]):
        super().__init__(data, max, float("-inf"))


# ─────────────────────────────────────────────
# Segment tree that also tracks the index of min/max
# Useful for "find leftmost/rightmost min" problems
# ─────────────────────────────────────────────
class MinIdxTree:
    """Stores (value, index) pairs; queries return (min_value, leftmost_index)."""

    def __init__(self, data: List[int]):
        self.n = len(data)
        INF = float("inf")
        self.tree: List[tuple] = [(INF, -1)] * (2 * self.n)
        for i, v in enumerate(data):
            self.tree[self.n + i] = (v, i)
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, i: int, val: int) -> None:
        i += self.n
        self.tree[i] = (val, i - self.n)
        while i > 1:
            i >>= 1
            self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])

    def query(self, l: int, r: int) -> tuple:
        """Returns (min_value, leftmost_index) over [l, r] inclusive."""
        lo, hi = (float("inf"), -1), (float("inf"), -1)
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                lo = min(lo, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                hi = min(hi, self.tree[r])
            l >>= 1
            r >>= 1
        return min(lo, hi)


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11]

    st = SumTree(arr)
    assert st.query(1, 3) == 15          # 3+5+7
    st.update(2, 10)                      # arr[2] = 10
    assert st.query(1, 3) == 20          # 3+10+7

    mn = MinTree(arr)
    assert mn.query(0, 4) == 1           # min of [1,3,5,7,9]

    mx = MaxTree(arr)
    assert mx.query(0, 5) == 11

    mi = MinIdxTree(arr)
    assert mi.query(1, 5) == (3, 1)      # min=3 at index 1

    print("All segment tree tests passed.")
