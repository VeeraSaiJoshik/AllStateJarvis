"""
NAME: AVL Tree / Balanced BST (and SortedList Alternative)
TAGS: AVL tree, balanced BST, sorted list, order statistics, data structures
DESCRIPTION: Self-balancing AVL tree supporting insert/delete/search in O(log n) with
    guaranteed height ≤ 1.44 log n. Also provides an order-statistics interface (k-th smallest,
    rank). In Python competitions, use sortedcontainers.SortedList as a faster drop-in replacement.
COMPLEXITY: O(log n) insert/delete/search/rank/kth, Space O(n)
"""

from __future__ import annotations
from typing import Optional, List


# ─────────────────────────────────────────────
# AVL Tree
# ─────────────────────────────────────────────
class AVLNode:
    __slots__ = ("key", "height", "size", "left", "right")

    def __init__(self, key: int):
        self.key = key
        self.height = 1
        self.size = 1
        self.left: Optional[AVLNode] = None
        self.right: Optional[AVLNode] = None


def _h(node: Optional[AVLNode]) -> int:
    return node.height if node else 0

def _sz(node: Optional[AVLNode]) -> int:
    return node.size if node else 0

def _update(node: AVLNode) -> None:
    node.height = 1 + max(_h(node.left), _h(node.right))
    node.size = 1 + _sz(node.left) + _sz(node.right)

def _balance(node: AVLNode) -> int:
    return _h(node.left) - _h(node.right)

def _rotate_right(y: AVLNode) -> AVLNode:
    x = y.left
    t = x.right
    x.right = y
    y.left = t
    _update(y)
    _update(x)
    return x

def _rotate_left(x: AVLNode) -> AVLNode:
    y = x.right
    t = y.left
    y.left = x
    x.right = t
    _update(x)
    _update(y)
    return y

def _rebalance(node: AVLNode) -> AVLNode:
    _update(node)
    bf = _balance(node)
    if bf > 1:
        if _balance(node.left) < 0:
            node.left = _rotate_left(node.left)
        return _rotate_right(node)
    if bf < -1:
        if _balance(node.right) > 0:
            node.right = _rotate_right(node.right)
        return _rotate_left(node)
    return node

def _insert(node: Optional[AVLNode], key: int) -> AVLNode:
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = _insert(node.left, key)
    elif key > node.key:
        node.right = _insert(node.right, key)
    else:
        return node   # duplicate — no-op (modify for multiset)
    return _rebalance(node)

def _min_node(node: AVLNode) -> AVLNode:
    while node.left:
        node = node.left
    return node

def _delete(node: Optional[AVLNode], key: int) -> Optional[AVLNode]:
    if node is None:
        return None
    if key < node.key:
        node.left = _delete(node.left, key)
    elif key > node.key:
        node.right = _delete(node.right, key)
    else:
        if node.left is None:
            return node.right
        if node.right is None:
            return node.left
        # Replace with in-order successor
        succ = _min_node(node.right)
        node.key = succ.key
        node.right = _delete(node.right, succ.key)
    return _rebalance(node)

def _search(node: Optional[AVLNode], key: int) -> bool:
    while node:
        if key == node.key:
            return True
        node = node.left if key < node.key else node.right
    return False

def _kth(node: Optional[AVLNode], k: int) -> int:
    """Return k-th smallest (1-indexed)."""
    while node:
        left_sz = _sz(node.left)
        if k == left_sz + 1:
            return node.key
        elif k <= left_sz:
            node = node.left
        else:
            k -= left_sz + 1
            node = node.right
    raise IndexError("k out of range")

def _rank(node: Optional[AVLNode], key: int) -> int:
    """Number of elements strictly less than key."""
    rank = 0
    while node:
        if key <= node.key:
            node = node.left
        else:
            rank += _sz(node.left) + 1
            node = node.right
    return rank


class AVLTree:
    """Order-statistics AVL tree. Supports duplicates if you modify _insert."""

    def __init__(self):
        self.root: Optional[AVLNode] = None

    def insert(self, key: int) -> None:
        self.root = _insert(self.root, key)

    def delete(self, key: int) -> None:
        self.root = _delete(self.root, key)

    def search(self, key: int) -> bool:
        return _search(self.root, key)

    def kth_smallest(self, k: int) -> int:
        """1-indexed k-th smallest element."""
        return _kth(self.root, k)

    def rank(self, key: int) -> int:
        """Number of elements strictly less than key (0-indexed rank)."""
        return _rank(self.root, key)

    def __len__(self) -> int:
        return _sz(self.root)

    def min_val(self) -> Optional[int]:
        if not self.root:
            return None
        return _min_node(self.root).key

    def max_val(self) -> Optional[int]:
        node = self.root
        if not node:
            return None
        while node.right:
            node = node.right
        return node.key

    def inorder(self) -> List[int]:
        result: List[int] = []
        def _inorder(node: Optional[AVLNode]) -> None:
            if node:
                _inorder(node.left)
                result.append(node.key)
                _inorder(node.right)
        _inorder(self.root)
        return result


# ─────────────────────────────────────────────
# SortedList wrapper (sortedcontainers)
# Fastest in Python competitions — O(sqrt(n)) amortized per op
# ─────────────────────────────────────────────
try:
    from sortedcontainers import SortedList

    class OrderedSet(SortedList):
        """
        Drop-in for AVLTree using SortedList.
        sl = OrderedSet()
        sl.add(x)        # insert
        sl.remove(x)     # delete
        sl[k]            # k-th smallest (0-indexed)
        sl.bisect_left(x) # rank of x (count of elements < x)
        """
        pass

except ImportError:
    pass   # sortedcontainers not available, use AVLTree above


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    avl = AVLTree()
    for v in [5, 3, 7, 1, 4, 6, 8, 2]:
        avl.insert(v)

    assert avl.inorder() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert len(avl) == 8
    assert avl.min_val() == 1
    assert avl.max_val() == 8
    assert avl.kth_smallest(1) == 1
    assert avl.kth_smallest(4) == 4
    assert avl.kth_smallest(8) == 8
    assert avl.rank(5) == 4   # [1,2,3,4] are < 5
    assert avl.search(6)
    assert not avl.search(9)

    avl.delete(5)
    assert avl.inorder() == [1, 2, 3, 4, 6, 7, 8]
    assert not avl.search(5)

    # Height invariant: must be <= 1.44 * log2(8) ≈ 4.3
    assert _h(avl.root) <= 5

    print("All AVL tree tests passed.")
