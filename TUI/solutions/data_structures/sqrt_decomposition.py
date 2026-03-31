"""
NAME: Square Root Decomposition and Mo's Algorithm
TAGS: sqrt-decomposition, range-queries, offline, mo-algorithm
DESCRIPTION: Split array into blocks of size sqrt(n) for O(sqrt(n)) range queries
             without a segment tree. Mo's algorithm processes offline range queries
             in O((n + q) * sqrt(n)) by sorting queries to minimize pointer movement.
             Use when segment trees are overkill or when offline processing is acceptable.
COMPLEXITY: Build O(n), Query/Update O(sqrt(n)); Mo's O((n+q)*sqrt(n))
"""

from typing import List, Tuple, Callable
import math


# ─── Sqrt Decomposition: Range Sum Query with Point Update ────────────────────

class SqrtRangeSum:
    """
    Range sum query and point update in O(sqrt(n)) each.
    """
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.block_size = max(1, int(math.isqrt(self.n)))
        self.arr = arr[:]
        num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.blocks = [0] * num_blocks

        for i, val in enumerate(self.arr):
            self.blocks[i // self.block_size] += val

    def update(self, i: int, val: int) -> None:
        """Set arr[i] = val in O(sqrt(n))."""
        self.blocks[i // self.block_size] += val - self.arr[i]
        self.arr[i] = val

    def query(self, lo: int, hi: int) -> int:
        """Sum of arr[lo..hi] inclusive in O(sqrt(n))."""
        result = 0
        block_lo = lo // self.block_size
        block_hi = hi // self.block_size

        if block_lo == block_hi:
            # Same block: iterate directly
            for i in range(lo, hi + 1):
                result += self.arr[i]
        else:
            # Left partial block
            for i in range(lo, (block_lo + 1) * self.block_size):
                result += self.arr[i]
            # Full middle blocks
            for b in range(block_lo + 1, block_hi):
                result += self.blocks[b]
            # Right partial block
            for i in range(block_hi * self.block_size, hi + 1):
                result += self.arr[i]

        return result

# Example:
# sr = SqrtRangeSum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# sr.query(0, 9) -> 55
# sr.update(4, 10); sr.query(0, 9) -> 60


# ─── Sqrt Decomposition: Range Minimum Query ─────────────────────────────────

class SqrtRangeMin:
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.block_size = max(1, int(math.isqrt(self.n)))
        self.arr = arr[:]
        num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.blocks = [float('inf')] * num_blocks

        for i, val in enumerate(self.arr):
            b = i // self.block_size
            self.blocks[b] = min(self.blocks[b], val)

    def update(self, i: int, val: int) -> None:
        """Update arr[i] = val. Rebuild affected block."""
        self.arr[i] = val
        b = i // self.block_size
        lo = b * self.block_size
        hi = min(lo + self.block_size, self.n)
        self.blocks[b] = min(self.arr[lo:hi])

    def query(self, lo: int, hi: int) -> int:
        result = float('inf')
        block_lo = lo // self.block_size
        block_hi = hi // self.block_size

        if block_lo == block_hi:
            for i in range(lo, hi + 1):
                result = min(result, self.arr[i])
        else:
            for i in range(lo, (block_lo + 1) * self.block_size):
                result = min(result, self.arr[i])
            for b in range(block_lo + 1, block_hi):
                result = min(result, self.blocks[b])
            for i in range(block_hi * self.block_size, hi + 1):
                result = min(result, self.arr[i])

        return result


# ─── Sqrt Decomposition: Range Assignment + Range Sum (Lazy) ─────────────────

class SqrtLazyRangeSum:
    """
    Range assign (set all in [l, r] to value) + range sum.
    Uses lazy tag per block.
    """
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.block_size = max(1, int(math.isqrt(self.n)))
        self.arr = arr[:]
        num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.block_sum = [0] * num_blocks
        self.lazy = [None] * num_blocks  # None means no pending assignment

        for i, val in enumerate(self.arr):
            self.block_sum[i // self.block_size] += val

    def _block_start(self, b: int) -> int:
        return b * self.block_size

    def _block_end(self, b: int) -> int:
        return min((b + 1) * self.block_size - 1, self.n - 1)

    def _push_down(self, b: int) -> None:
        if self.lazy[b] is not None:
            val = self.lazy[b]
            for i in range(self._block_start(b), self._block_end(b) + 1):
                self.arr[i] = val
            self.lazy[b] = None

    def assign(self, lo: int, hi: int, val: int) -> None:
        """Set all elements in [lo, hi] to val."""
        block_lo = lo // self.block_size
        block_hi = hi // self.block_size

        if block_lo == block_hi:
            self._push_down(block_lo)
            for i in range(lo, hi + 1):
                self.arr[i] = val
            # Recompute block sum
            bs, be = self._block_start(block_lo), self._block_end(block_lo)
            self.block_sum[block_lo] = sum(self.arr[bs:be + 1])
        else:
            # Left partial block
            self._push_down(block_lo)
            for i in range(lo, self._block_end(block_lo) + 1):
                self.arr[i] = val
            bs, be = self._block_start(block_lo), self._block_end(block_lo)
            self.block_sum[block_lo] = sum(self.arr[bs:be + 1])

            # Full middle blocks (lazy)
            for b in range(block_lo + 1, block_hi):
                self.lazy[b] = val
                self.block_sum[b] = val * (self._block_end(b) - self._block_start(b) + 1)

            # Right partial block
            self._push_down(block_hi)
            for i in range(self._block_start(block_hi), hi + 1):
                self.arr[i] = val
            bs, be = self._block_start(block_hi), self._block_end(block_hi)
            self.block_sum[block_hi] = sum(self.arr[bs:be + 1])

    def query(self, lo: int, hi: int) -> int:
        result = 0
        block_lo = lo // self.block_size
        block_hi = hi // self.block_size

        if block_lo == block_hi:
            self._push_down(block_lo)
            result = sum(self.arr[lo:hi + 1])
        else:
            self._push_down(block_lo)
            result += sum(self.arr[lo:self._block_end(block_lo) + 1])
            for b in range(block_lo + 1, block_hi):
                result += self.block_sum[b]
            self._push_down(block_hi)
            result += sum(self.arr[self._block_start(block_hi):hi + 1])

        return result


# ─── Mo's Algorithm ──────────────────────────────────────────────────────────
# Offline range queries sorted to minimize pointer movements.
# Block-sort: sort by (block of l, r) with alternating r direction for even blocks.

def mo_algorithm(arr: List[int], queries: List[Tuple[int, int]]) -> List[int]:
    """
    Solve range frequency/count queries offline.
    Returns answers[i] = number of distinct elements in arr[queries[i][0]..queries[i][1]].

    Customize add/remove/answer functions for different query types.
    """
    n = len(arr)
    q = len(queries)
    if n == 0 or q == 0:
        return []

    block_size = max(1, int(math.isqrt(n)))

    # Attach original index for output ordering
    indexed_queries = [(l, r, i) for i, (l, r) in enumerate(queries)]

    # Mo's ordering: sort by (block of l, r ASC if even block, r DESC if odd block)
    indexed_queries.sort(key=lambda x: (
        x[0] // block_size,
        x[1] if (x[0] // block_size) % 2 == 0 else -x[1]
    ))

    # ── State for "count distinct" query ──────────────────────────────────────
    from collections import defaultdict
    freq = defaultdict(int)
    distinct = 0

    def add(pos: int) -> None:
        nonlocal distinct
        freq[arr[pos]] += 1
        if freq[arr[pos]] == 1:
            distinct += 1

    def remove(pos: int) -> None:
        nonlocal distinct
        freq[arr[pos]] -= 1
        if freq[arr[pos]] == 0:
            distinct -= 1

    answers = [0] * q
    cur_l, cur_r = 0, -1

    for l, r, qi in indexed_queries:
        # Expand / shrink window to [l, r]
        while cur_r < r:
            cur_r += 1; add(cur_r)
        while cur_l > l:
            cur_l -= 1; add(cur_l)
        while cur_r > r:
            remove(cur_r); cur_r -= 1
        while cur_l < l:
            remove(cur_l); cur_l += 1

        answers[qi] = distinct

    return answers

# Example:
# arr = [1, 2, 1, 3, 2]
# queries = [(0, 4), (1, 3), (2, 4)]
# answers -> [3, 3, 3]  (all three windows contain 3 distinct values)


# ─── Mo's Algorithm Template (customizable) ───────────────────────────────────
class MoSolver:
    """
    Generic Mo's algorithm solver. Override add/remove/answer.
    """
    def __init__(self, arr: List[int], queries: List[Tuple[int, int]]):
        self.arr = arr
        self.queries = queries
        self.n = len(arr)
        self.q = len(queries)
        self.block_size = max(1, int(math.isqrt(self.n)))

    def _sort_queries(self):
        return sorted(
            enumerate(self.queries),
            key=lambda x: (
                x[1][0] // self.block_size,
                x[1][1] if (x[1][0] // self.block_size) % 2 == 0 else -x[1][1]
            )
        )

    def add(self, pos: int) -> None:
        raise NotImplementedError

    def remove(self, pos: int) -> None:
        raise NotImplementedError

    def answer(self) -> int:
        raise NotImplementedError

    def solve(self) -> List[int]:
        results = [0] * self.q
        cur_l, cur_r = 0, -1

        for qi, (l, r) in self._sort_queries():
            while cur_r < r: cur_r += 1; self.add(cur_r)
            while cur_l > l: cur_l -= 1; self.add(cur_l)
            while cur_r > r: self.remove(cur_r); cur_r -= 1
            while cur_l < l: self.remove(cur_l); cur_l += 1
            results[qi] = self.answer()

        return results


if __name__ == "__main__":
    sr = SqrtRangeSum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sr.query(0, 9) == 55
    sr.update(4, 10)
    assert sr.query(0, 9) == 60
    assert sr.query(0, 3) == 10  # 1+2+3+4

    sm = SqrtRangeMin([5, 3, 1, 4, 2])
    assert sm.query(0, 4) == 1
    assert sm.query(0, 2) == 1
    assert sm.query(3, 4) == 2
    sm.update(2, 10)
    assert sm.query(0, 4) == 2

    # Mo's distinct count
    arr = [1, 2, 1, 3, 2]
    queries = [(0, 4), (1, 3), (2, 4)]
    answers = mo_algorithm(arr, queries)
    assert answers == [3, 3, 3]

    print("All tests passed.")
