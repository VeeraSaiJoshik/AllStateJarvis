"""
NAME: Deque-Based BFS Patterns and Circular Buffer
TAGS: deque, bfs, 0-1-bfs, circular-buffer, graph
DESCRIPTION: Advanced BFS patterns using deque: 0-1 BFS for graphs with 0/1 edge weights,
             multi-source BFS, level-order traversal, and a circular ring buffer implementation.
             Use 0-1 BFS instead of Dijkstra when edge weights are only 0 or 1 (O(V+E) vs O((V+E)logV)).
COMPLEXITY: BFS O(V+E), 0-1 BFS O(V+E), CircularBuffer O(1) per operation
"""

from collections import deque
from typing import List, Optional, Tuple, Iterator
import math


# ─── Standard BFS ─────────────────────────────────────────────────────────────

def bfs_shortest_path(graph: List[List[int]], src: int, dst: int) -> int:
    """Shortest path (by hop count) from src to dst. Returns -1 if unreachable."""
    n = len(graph)
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])

    while q:
        node = q.popleft()
        if node == dst:
            return dist[node]
        for neighbor in graph[node]:
            if dist[neighbor] == -1:
                dist[neighbor] = dist[node] + 1
                q.append(neighbor)

    return -1

# Example:
# graph = [[1,2],[0,3],[0,4],[1],[2]]
# bfs_shortest_path(graph, 0, 4) -> 2


# ─── BFS on 2D Grid ───────────────────────────────────────────────────────────

def bfs_grid(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """
    Shortest path in a 2D grid from start to end.
    grid[r][c] = 0 is passable, 1 is wall.
    Returns -1 if unreachable.
    """
    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    er, ec = end

    if grid[sr][sc] == 1 or grid[er][ec] == 1:
        return -1

    dist = [[-1] * cols for _ in range(rows)]
    dist[sr][sc] = 0
    q = deque([(sr, sc)])
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while q:
        r, c = q.popleft()
        if r == er and c == ec:
            return dist[r][c]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))

    return -1


# ─── Multi-Source BFS ─────────────────────────────────────────────────────────
# Initialize BFS with multiple sources simultaneously.
# Classic use: 0-1 matrix (distance to nearest 0), rotten oranges.

def multi_source_bfs(grid: List[List[int]]) -> List[List[int]]:
    """
    Returns distance of each cell to the nearest cell with value 0.
    -1 for cells that cannot reach any 0.
    """
    rows, cols = len(grid), len(grid[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    q = deque()

    # Seed queue with all 0-cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                dist[r][c] = 0
                q.append((r, c))

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == float('inf'):
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))

    return [[d if d != float('inf') else -1 for d in row] for row in dist]

# Example:
# grid = [[0,0,0],[0,1,0],[1,1,1]]
# result -> [[0,0,0],[0,1,0],[1,2,1]]


# ─── 0-1 BFS ──────────────────────────────────────────────────────────────────
# For graphs with edge weights 0 or 1. Use deque: 0-weight to front, 1-weight to back.
# Equivalent to Dijkstra but O(V+E) instead of O((V+E)logV).

def zero_one_bfs(n: int, edges: List[Tuple[int, int, int]], src: int) -> List[int]:
    """
    Single-source shortest path with 0/1 edge weights.
    edges: list of (u, v, w) where w is 0 or 1.
    Returns dist[v] = shortest distance from src to v.
    """
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    dist = [float('inf')] * n
    dist[src] = 0
    q = deque([src])

    while q:
        u = q.popleft()
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    q.appendleft(v)  # 0-cost: add to front
                else:
                    q.append(v)      # 1-cost: add to back

    return dist

# Example:
# n = 5, edges = [(0,1,1),(0,2,0),(1,3,1),(2,3,0),(3,4,1)], src = 0
# dist -> [0, 1, 0, 0, 1]


# ─── 0-1 BFS on Grid ─────────────────────────────────────────────────────────
# Minimum flips to create a path from top-left to bottom-right.
# Moving within same value = cost 0, flipping = cost 1.

def min_flips_path(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[0][0] = 0
    q = deque([(0, 0)])
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cost = 0 if grid[nr][nc] == grid[r][c] else 1
                new_dist = dist[r][c] + cost
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    if cost == 0:
                        q.appendleft((nr, nc))
                    else:
                        q.append((nr, nc))

    return dist[rows - 1][cols - 1]


# ─── BFS Level Order Traversal (Binary Tree) ─────────────────────────────────

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []

    result = []
    q = deque([root])

    while q:
        level = []
        for _ in range(len(q)):  # process exactly this level
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)

    return result

# Example:
# Tree: 3 -> {9, 20 -> {15, 7}}
# result -> [[3], [9, 20], [15, 7]]


# ─── Zigzag Level Order (Deque-based alternation) ─────────────────────────────

def zigzag_level_order(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []

    result = []
    q = deque([root])
    left_to_right = True

    while q:
        level_dq = deque()
        for _ in range(len(q)):
            node = q.popleft()
            if left_to_right:
                level_dq.append(node.val)
            else:
                level_dq.appendleft(node.val)  # reverse by inserting at front
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(list(level_dq))
        left_to_right = not left_to_right

    return result


# ─── Circular Buffer (Ring Buffer) ────────────────────────────────────────────
# Fixed-capacity FIFO queue with O(1) enqueue and dequeue.
# No dynamic resizing. Useful for sliding windows and streaming algorithms.

class CircularBuffer:
    """
    Fixed-size circular ring buffer.
    Supports enqueue, dequeue, peek_front, peek_rear in O(1).
    """
    def __init__(self, capacity: int):
        self._buf = [None] * capacity
        self._cap = capacity
        self._head = 0   # index of the front element
        self._tail = 0   # index where next element will be written
        self._size = 0

    def enqueue(self, val) -> bool:
        if self._size == self._cap:
            return False  # full
        self._buf[self._tail] = val
        self._tail = (self._tail + 1) % self._cap
        self._size += 1
        return True

    def dequeue(self):
        if self._size == 0:
            raise IndexError("dequeue from empty buffer")
        val = self._buf[self._head]
        self._buf[self._head] = None  # help GC
        self._head = (self._head + 1) % self._cap
        self._size -= 1
        return val

    def peek_front(self):
        if self._size == 0:
            raise IndexError("buffer is empty")
        return self._buf[self._head]

    def peek_rear(self):
        if self._size == 0:
            raise IndexError("buffer is empty")
        return self._buf[(self._tail - 1) % self._cap]

    def is_empty(self) -> bool:
        return self._size == 0

    def is_full(self) -> bool:
        return self._size == self._cap

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator:
        """Iterate from front to rear."""
        for i in range(self._size):
            yield self._buf[(self._head + i) % self._cap]

    def __repr__(self) -> str:
        return f"CircularBuffer({list(self)})"


# ─── Circular Buffer with Overwrite (for sliding window log) ─────────────────

class OverwriteCircularBuffer:
    """
    Circular buffer that overwrites oldest element when full.
    Useful for maintaining the last N events/values.
    """
    def __init__(self, capacity: int):
        self._buf = [None] * capacity
        self._cap = capacity
        self._head = 0
        self._size = 0

    def push(self, val) -> None:
        self._buf[self._head] = val
        self._head = (self._head + 1) % self._cap
        if self._size < self._cap:
            self._size += 1

    def __getitem__(self, idx: int):
        """0 = oldest, size-1 = newest."""
        if idx < 0 or idx >= self._size:
            raise IndexError(idx)
        start = (self._head - self._size) % self._cap
        return self._buf[(start + idx) % self._cap]

    def __len__(self) -> int:
        return self._size

    def to_list(self) -> List:
        return [self[i] for i in range(self._size)]


# ─── Deque as Stack + Queue simultaneously ────────────────────────────────────
# Python's deque supports O(1) ops on both ends natively.
# This section just documents the idioms:
#
# Stack (LIFO): append() + pop()
# Queue (FIFO): append() + popleft()
# Priority (simulate): appendleft() to push high-priority items
#
# Pattern: sliding window using deque as a monotonic structure
# (see monotonic_deque.py for full coverage)


# ─── Word Ladder (BFS on word graph) ─────────────────────────────────────────
# Classic BFS on implicit graph: each transformation is one hop.

def word_ladder_length(begin_word: str, end_word: str, word_list: List[str]) -> int:
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    q = deque([(begin_word, 1)])
    visited = {begin_word}

    while q:
        word, steps = q.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i + 1:]
                if new_word == end_word:
                    return steps + 1
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    q.append((new_word, steps + 1))

    return 0

# Example:
# begin_word = "hit", end_word = "cog"
# word_list = ["hot","dot","dog","lot","log","cog"]
# answer -> 5  (hit->hot->dot->dog->cog)


if __name__ == "__main__":
    # BFS graph
    graph = [[1, 2], [0, 3], [0, 4], [1], [2]]
    assert bfs_shortest_path(graph, 0, 4) == 2

    # Multi-source BFS
    grid = [[0,0,0],[0,1,0],[1,1,1]]
    result = multi_source_bfs(grid)
    assert result == [[0,0,0],[0,1,0],[1,2,1]]

    # 0-1 BFS
    edges = [(0,1,1),(0,2,0),(1,3,1),(2,3,0),(3,4,1)]
    dist = zero_one_bfs(5, edges, 0)
    assert dist == [0, 1, 0, 0, 1]

    # Circular buffer
    cb = CircularBuffer(3)
    assert cb.enqueue(1) and cb.enqueue(2) and cb.enqueue(3)
    assert not cb.enqueue(4)  # full
    assert cb.dequeue() == 1
    assert cb.enqueue(4)
    assert list(cb) == [2, 3, 4]

    # Overwrite buffer
    ob = OverwriteCircularBuffer(3)
    for v in [1, 2, 3, 4, 5]:
        ob.push(v)
    assert ob.to_list() == [3, 4, 5]

    # Word ladder
    assert word_ladder_length("hit", "cog", ["hot","dot","dog","lot","log","cog"]) == 5

    print("All tests passed.")
