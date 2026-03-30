"""
NAME: Trie (Prefix Tree) and XOR Trie
TAGS: trie, prefix tree, string, XOR, maximum XOR, bitwise, data structures
DESCRIPTION: Standard Trie supports insert/search/prefix-count operations on strings
    in O(L) where L is string length. XOR Trie inserts integers as 30-bit binary strings
    and finds the number that maximizes XOR with a query — essential for XOR maximization problems.
COMPLEXITY: Trie: O(L) per op, Space O(n*L*ALPHA); XOR Trie: O(30) per op, Space O(n*30)
"""

from typing import List, Optional


# ─────────────────────────────────────────────
# Standard Trie — Strings
# ─────────────────────────────────────────────
class TrieNode:
    __slots__ = ("children", "is_end", "count", "prefix_count")

    def __init__(self):
        self.children: dict = {}
        self.is_end: bool = False
        self.count: int = 0          # how many times this exact word was inserted
        self.prefix_count: int = 0   # how many words pass through this node


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert word into trie."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.prefix_count += 1
        node.is_end = True
        node.count += 1

    def search(self, word: str) -> bool:
        """Return True if word was inserted."""
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Return True if any inserted word starts with prefix."""
        return self._find(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """Count how many inserted words start with prefix."""
        node = self._find(prefix)
        return node.prefix_count if node else 0

    def delete(self, word: str) -> bool:
        """Remove one occurrence of word. Returns True if found."""
        def _del(node: TrieNode, word: str, depth: int) -> bool:
            if depth == len(word):
                if not node.is_end:
                    return False
                node.count -= 1
                if node.count == 0:
                    node.is_end = False
                return True
            ch = word[depth]
            if ch not in node.children:
                return False
            child = node.children[ch]
            found = _del(child, word, depth + 1)
            if found:
                child.prefix_count -= 1
                if child.prefix_count == 0:
                    del node.children[ch]
            return found

        return _del(self.root, word, 0)

    def _find(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def all_words_with_prefix(self, prefix: str) -> List[str]:
        """Return all words that start with prefix."""
        node = self._find(prefix)
        if not node:
            return []
        result: List[str] = []
        self._dfs(node, list(prefix), result)
        return result

    def _dfs(self, node: TrieNode, path: List[str], result: List[str]) -> None:
        if node.is_end:
            result.append("".join(path))
        for ch, child in node.children.items():
            path.append(ch)
            self._dfs(child, path, result)
            path.pop()


# ─────────────────────────────────────────────
# XOR Trie — Maximum XOR
# ─────────────────────────────────────────────
class XorTrieNode:
    __slots__ = ("children", "count")

    def __init__(self):
        self.children: List[Optional["XorTrieNode"]] = [None, None]
        self.count: int = 0


class XorTrie:
    """
    Insert integers and query for maximum XOR with a given value.
    Supports deletion via count tracking.
    Bit depth = 30 covers all numbers up to 10^9.
    """

    BITS = 30

    def __init__(self):
        self.root = XorTrieNode()

    def insert(self, num: int, delta: int = 1) -> None:
        """Insert (delta=1) or delete (delta=-1) num from trie."""
        node = self.root
        for i in range(self.BITS, -1, -1):
            bit = (num >> i) & 1
            if node.children[bit] is None:
                node.children[bit] = XorTrieNode()
            node = node.children[bit]
            node.count += delta

    def delete(self, num: int) -> None:
        self.insert(num, -1)

    def max_xor(self, num: int) -> int:
        """Return max XOR of num with any number in the trie."""
        node = self.root
        result = 0
        for i in range(self.BITS, -1, -1):
            bit = (num >> i) & 1
            want = 1 - bit           # we want the opposite bit to maximize XOR
            # Check if 'want' subtree is non-empty
            if (node.children[want] is not None and
                    node.children[want].count > 0):
                result |= (1 << i)
                node = node.children[want]
            elif node.children[bit] is not None:
                node = node.children[bit]
            else:
                break
        return result

    def min_xor(self, num: int) -> int:
        """Return min XOR of num with any number in the trie."""
        node = self.root
        result = 0
        for i in range(self.BITS, -1, -1):
            bit = (num >> i) & 1
            # prefer same bit to minimize XOR
            if (node.children[bit] is not None and
                    node.children[bit].count > 0):
                node = node.children[bit]
            elif node.children[1 - bit] is not None:
                result |= (1 << i)
                node = node.children[1 - bit]
            else:
                break
        return result


# ─────────────────────────────────────────────
# Array-based XOR Trie (faster, fixed-size)
# ─────────────────────────────────────────────
class XorTrieFast:
    """
    Array-based XOR Trie — faster due to no object overhead.
    Pre-allocates MAXN * 2 nodes. Adjust MAXN for problem constraints.
    """

    BITS = 30
    MAXN = 100001 * (BITS + 1)   # max nodes

    def __init__(self):
        self.ch = [[-1, -1] for _ in range(self.MAXN)]
        self.cnt = [0] * self.MAXN
        self.sz = 1   # root = node 0

    def insert(self, num: int) -> None:
        node = 0
        for i in range(self.BITS, -1, -1):
            bit = (num >> i) & 1
            if self.ch[node][bit] == -1:
                self.ch[node][bit] = self.sz
                self.sz += 1
            node = self.ch[node][bit]
            self.cnt[node] += 1

    def max_xor(self, num: int) -> int:
        node = 0
        result = 0
        for i in range(self.BITS, -1, -1):
            bit = (num >> i) & 1
            want = 1 - bit
            nb = self.ch[node][want]
            if nb != -1 and self.cnt[nb] > 0:
                result |= (1 << i)
                node = nb
            elif self.ch[node][bit] != -1:
                node = self.ch[node][bit]
            else:
                break
        return result


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Standard Trie
    t = Trie()
    for w in ["apple", "app", "application", "apply", "banana"]:
        t.insert(w)
    assert t.search("apple")
    assert not t.search("ap")
    assert t.starts_with("app")
    assert t.count_prefix("app") == 4   # apple, app, application, apply
    t.delete("apple")
    assert not t.search("apple")
    assert t.count_prefix("app") == 3

    words = sorted(t.all_words_with_prefix("app"))
    assert words == ["app", "application", "apply"], words

    # XOR Trie
    xt = XorTrie()
    for n in [3, 10, 5, 25, 2, 8]:
        xt.insert(n)
    assert xt.max_xor(5) == 28          # 5 XOR 25 = 28

    xt.delete(25)
    assert xt.max_xor(5) == 15          # 5 XOR 10 = 15

    print("All Trie tests passed.")
