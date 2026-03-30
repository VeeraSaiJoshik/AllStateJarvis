"""
NAME: Binary Search Tree — BST, Treap, and Randomized BST
TAGS: BST, treap, randomized BST, order statistics, split, merge, data structures
DESCRIPTION: Standard BST for reference, plus a Treap (BST + heap on random priorities)
    that gives O(log n) expected ops without rebalancing. Treap's split/merge interface
    enables powerful operations like insert/delete/kth/rank/reverse in competitions.
COMPLEXITY: Treap: O(log n) expected per op, Space O(n); worst-case BST: O(n)
"""

from __future__ import annotations
import random
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────
# Plain BST (reference / interview template)
# ─────────────────────────────────────────────
class BSTNode:
    __slots__ = ("key", "left", "right")

    def __init__(self, key: int):
        self.key = key
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None


class BST:
    """Unbalanced BST — O(log n) average, O(n) worst case."""

    def __init__(self):
        self.root: Optional[BSTNode] = None

    def insert(self, key: int) -> None:
        def _ins(node: Optional[BSTNode], key: int) -> BSTNode:
            if node is None:
                return BSTNode(key)
            if key < node.key:
                node.left = _ins(node.left, key)
            elif key > node.key:
                node.right = _ins(node.right, key)
            return node
        self.root = _ins(self.root, key)

    def search(self, key: int) -> bool:
        node = self.root
        while node:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False

    def delete(self, key: int) -> None:
        def _del(node: Optional[BSTNode], key: int) -> Optional[BSTNode]:
            if node is None:
                return None
            if key < node.key:
                node.left = _del(node.left, key)
            elif key > node.key:
                node.right = _del(node.right, key)
            else:
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                # In-order successor
                succ = node.right
                while succ.left:
                    succ = succ.left
                node.key = succ.key
                node.right = _del(node.right, succ.key)
            return node
        self.root = _del(self.root, key)

    def inorder(self) -> List[int]:
        result: List[int] = []
        stack: List[BSTNode] = []
        node = self.root
        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            result.append(node.key)
            node = node.right
        return result

    def floor(self, key: int) -> Optional[int]:
        """Largest element <= key."""
        res = None
        node = self.root
        while node:
            if node.key == key:
                return key
            elif node.key < key:
                res = node.key
                node = node.right
            else:
                node = node.left
        return res

    def ceil(self, key: int) -> Optional[int]:
        """Smallest element >= key."""
        res = None
        node = self.root
        while node:
            if node.key == key:
                return key
            elif node.key > key:
                res = node.key
                node = node.left
            else:
                node = node.right
        return res


# ─────────────────────────────────────────────
# Treap — Randomized BST
# Split/merge paradigm; supports all BST ops + implicit treap for sequences
# ─────────────────────────────────────────────
class TreapNode:
    __slots__ = ("key", "priority", "size", "left", "right")

    def __init__(self, key: int):
        self.key = key
        self.priority = random.random()
        self.size = 1
        self.left: Optional[TreapNode] = None
        self.right: Optional[TreapNode] = None


def _tsz(node: Optional[TreapNode]) -> int:
    return node.size if node else 0

def _tupdate(node: TreapNode) -> None:
    node.size = 1 + _tsz(node.left) + _tsz(node.right)

def _split(node: Optional[TreapNode], key: int) -> Tuple[Optional[TreapNode], Optional[TreapNode]]:
    """Split into (< key) and (>= key)."""
    if node is None:
        return None, None
    if node.key < key:
        l, r = _split(node.right, key)
        node.right = l
        _tupdate(node)
        return node, r
    else:
        l, r = _split(node.left, key)
        node.left = r
        _tupdate(node)
        return l, node

def _merge(left: Optional[TreapNode], right: Optional[TreapNode]) -> Optional[TreapNode]:
    """Merge two treaps where all keys in left < all keys in right."""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        left.right = _merge(left.right, right)
        _tupdate(left)
        return left
    else:
        right.left = _merge(left, right.left)
        _tupdate(right)
        return right

def _tkth(node: Optional[TreapNode], k: int) -> int:
    """1-indexed k-th smallest."""
    while node:
        lsz = _tsz(node.left)
        if k == lsz + 1:
            return node.key
        elif k <= lsz:
            node = node.left
        else:
            k -= lsz + 1
            node = node.right
    raise IndexError("k out of range")

def _trank(node: Optional[TreapNode], key: int) -> int:
    """Count of elements < key."""
    rank = 0
    while node:
        if key <= node.key:
            node = node.left
        else:
            rank += _tsz(node.left) + 1
            node = node.right
    return rank


class Treap:
    """
    Treap with split/merge. Supports:
      insert, delete, search, kth_smallest, rank, floor, ceil.
    Use for order-statistics + custom operations in competitions.
    """

    def __init__(self):
        self.root: Optional[TreapNode] = None

    def insert(self, key: int) -> None:
        if self.search(key):
            return   # no duplicates; remove this check for multiset
        l, r = _split(self.root, key)
        self.root = _merge(_merge(l, TreapNode(key)), r)

    def delete(self, key: int) -> None:
        l, r = _split(self.root, key)
        _, r = _split(r, key + 1)
        self.root = _merge(l, r)

    def search(self, key: int) -> bool:
        node = self.root
        while node:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False

    def kth_smallest(self, k: int) -> int:
        """1-indexed."""
        return _tkth(self.root, k)

    def rank(self, key: int) -> int:
        """Count of elements < key."""
        return _trank(self.root, key)

    def floor(self, key: int) -> Optional[int]:
        """Largest element <= key."""
        l, r = _split(self.root, key + 1)
        res = None
        if l:
            res = _tkth(l, _tsz(l))
        self.root = _merge(l, r)
        return res

    def ceil(self, key: int) -> Optional[int]:
        """Smallest element >= key."""
        l, r = _split(self.root, key)
        res = None
        if r:
            res = _tkth(r, 1)
        self.root = _merge(l, r)
        return res

    def __len__(self) -> int:
        return _tsz(self.root)

    def inorder(self) -> List[int]:
        result: List[int] = []
        def _dfs(node: Optional[TreapNode]) -> None:
            if node:
                _dfs(node.left)
                result.append(node.key)
                _dfs(node.right)
        _dfs(self.root)
        return result


# ─────────────────────────────────────────────
# Implicit Treap — sequence with O(log n) split/merge/reverse
# Use for "rope" style problems: range reverse, range sum, etc.
# ─────────────────────────────────────────────
class ImplicitNode:
    __slots__ = ("val", "priority", "size", "lazy_rev", "left", "right")

    def __init__(self, val: int):
        self.val = val
        self.priority = random.random()
        self.size = 1
        self.lazy_rev = False
        self.left: Optional[ImplicitNode] = None
        self.right: Optional[ImplicitNode] = None


def _isz(t: Optional[ImplicitNode]) -> int:
    return t.size if t else 0

def _iupdate(t: ImplicitNode) -> None:
    t.size = 1 + _isz(t.left) + _isz(t.right)

def _push(t: Optional[ImplicitNode]) -> None:
    if t and t.lazy_rev:
        t.left, t.right = t.right, t.left
        if t.left:
            t.left.lazy_rev ^= True
        if t.right:
            t.right.lazy_rev ^= True
        t.lazy_rev = False

def _isplit(t: Optional[ImplicitNode], k: int):
    """Split into first k elements and the rest."""
    if t is None:
        return None, None
    _push(t)
    if _isz(t.left) >= k:
        l, r = _isplit(t.left, k)
        t.left = r
        _iupdate(t)
        return l, t
    else:
        l, r = _isplit(t.right, k - _isz(t.left) - 1)
        t.right = l
        _iupdate(t)
        return t, r

def _imerge(l: Optional[ImplicitNode], r: Optional[ImplicitNode]) -> Optional[ImplicitNode]:
    _push(l)
    _push(r)
    if l is None:
        return r
    if r is None:
        return l
    if l.priority > r.priority:
        l.right = _imerge(l.right, r)
        _iupdate(l)
        return l
    else:
        r.left = _imerge(l, r.left)
        _iupdate(r)
        return r


class ImplicitTreap:
    """
    Implicit treap — indexed sequence with O(log n):
      insert at position, delete at position, range reverse.
    """

    def __init__(self, data: Optional[List[int]] = None):
        self.root: Optional[ImplicitNode] = None
        if data:
            for v in data:
                self.root = _imerge(self.root, ImplicitNode(v))

    def insert(self, pos: int, val: int) -> None:
        """Insert val before position pos (0-indexed)."""
        l, r = _isplit(self.root, pos)
        self.root = _imerge(_imerge(l, ImplicitNode(val)), r)

    def delete(self, pos: int) -> None:
        """Delete element at position pos."""
        l, m = _isplit(self.root, pos)
        _, r = _isplit(m, 1)
        self.root = _imerge(l, r)

    def reverse(self, l: int, r: int) -> None:
        """Reverse subarray [l, r] (0-indexed, inclusive)."""
        a, b = _isplit(self.root, l)
        b, c = _isplit(b, r - l + 1)
        if b:
            b.lazy_rev ^= True
        self.root = _imerge(_imerge(a, b), c)

    def to_list(self) -> List[int]:
        result: List[int] = []
        def _dfs(node: Optional[ImplicitNode]) -> None:
            if node:
                _push(node)
                _dfs(node.left)
                result.append(node.val)
                _dfs(node.right)
        _dfs(self.root)
        return result

    def __len__(self) -> int:
        return _isz(self.root)


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)

    # BST
    bst = BST()
    for v in [5, 3, 7, 1, 4, 6, 8]:
        bst.insert(v)
    assert bst.inorder() == [1, 3, 4, 5, 6, 7, 8]
    assert bst.floor(5) == 5
    assert bst.floor(2) == 1
    assert bst.ceil(2) == 3
    bst.delete(5)
    assert 5 not in bst.inorder()

    # Treap
    t = Treap()
    for v in [5, 3, 7, 1, 4, 6, 8]:
        t.insert(v)
    assert t.inorder() == [1, 3, 4, 5, 6, 7, 8]
    assert t.kth_smallest(3) == 4
    assert t.rank(5) == 3
    assert t.floor(2) == 1
    assert t.ceil(2) == 3
    t.delete(5)
    assert not t.search(5)

    # Implicit Treap
    it = ImplicitTreap([1, 2, 3, 4, 5])
    it.reverse(1, 3)
    assert it.to_list() == [1, 4, 3, 2, 5]
    it.insert(2, 99)
    assert it.to_list() == [1, 4, 99, 3, 2, 5]
    it.delete(2)
    assert it.to_list() == [1, 4, 3, 2, 5]

    print("All BST / Treap tests passed.")
