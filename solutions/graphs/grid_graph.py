"""
NAME: Grid Graph — BFS/DFS, Flood Fill, Island Counting, Multi-Source BFS
TAGS: graph, grid, bfs, dfs, flood-fill, islands, multi-source, matrix, 2d
DESCRIPTION: Graph algorithms adapted for 2D grids. Covers shortest path on grids,
    flood fill (painting connected regions), connected component counting (islands),
    and multi-source BFS for distance-to-nearest-target problems. Handles 4-directional
    and 8-directional movement; essential for maze/map problems.
COMPLEXITY: BFS/DFS O(R*C), Space O(R*C)
"""

from collections import deque
from typing import List, Tuple, Optional, Callable


# ──────────────────────────────────────────────
# Direction constants
# ──────────────────────────────────────────────

DIRS_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]             # 4-directional
DIRS_8 = [(dr, dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc) != (0,0)]  # 8-directional


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


# ──────────────────────────────────────────────
# 1. BFS Shortest Path on Grid
#    Returns shortest distance from (sr, sc) to (er, ec).
#    Passable cells satisfy: passable(grid[r][c]) == True.
# ──────────────────────────────────────────────

def bfs_grid_shortest(grid: List[List], sr: int, sc: int,
                       er: int, ec: int,
                       passable: Callable = lambda x: x != '#',
                       dirs: List[Tuple] = DIRS_4) -> int:
    """
    BFS shortest path on grid from (sr,sc) to (er,ec).
    Returns distance, or -1 if unreachable.
    """
    rows, cols = len(grid), len(grid[0])
    if not passable(grid[sr][sc]) or not passable(grid[er][ec]):
        return -1
    visited = [[False] * cols for _ in range(rows)]
    visited[sr][sc] = True
    queue: deque = deque([(sr, sc, 0)])
    while queue:
        r, c, d = queue.popleft()
        if r == er and c == ec:
            return d
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and not visited[nr][nc] and passable(grid[nr][nc]):
                visited[nr][nc] = True
                queue.append((nr, nc, d + 1))
    return -1


def bfs_grid_distances(grid: List[List], sr: int, sc: int,
                        passable: Callable = lambda x: x != '#',
                        dirs: List[Tuple] = DIRS_4) -> List[List[int]]:
    """
    BFS from (sr, sc) — returns full distance matrix.
    dist[r][c] = shortest distance from source; -1 if unreachable.
    """
    rows, cols = len(grid), len(grid[0])
    dist = [[-1] * cols for _ in range(rows)]
    if not passable(grid[sr][sc]):
        return dist
    dist[sr][sc] = 0
    queue: deque = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and dist[nr][nc] == -1 and passable(grid[nr][nc]):
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return dist


# ──────────────────────────────────────────────
# 2. Multi-Source BFS on Grid
#    Start BFS simultaneously from all source cells.
#    Returns distance matrix where dist[r][c] = min dist to nearest source.
# ──────────────────────────────────────────────

def multi_source_bfs_grid(grid: List[List],
                           is_source: Callable = lambda x: x == 0,
                           passable: Callable = lambda x: True,
                           dirs: List[Tuple] = DIRS_4) -> List[List[int]]:
    """
    Multi-source BFS on grid.
    dist[r][c] = distance to nearest source cell; -1 if unreachable.
    Example use: 01-matrix problem, nearest 0, nearest gate.
    """
    rows, cols = len(grid), len(grid[0])
    dist = [[-1] * cols for _ in range(rows)]
    queue: deque = deque()

    for r in range(rows):
        for c in range(cols):
            if is_source(grid[r][c]):
                dist[r][c] = 0
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and dist[nr][nc] == -1 and passable(grid[nr][nc]):
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))

    return dist


# ──────────────────────────────────────────────
# 3. Flood Fill (DFS-based)
#    Fill all connected cells of the same value with a new value.
# ──────────────────────────────────────────────

def flood_fill(grid: List[List], sr: int, sc: int, new_color,
               dirs: List[Tuple] = DIRS_4) -> List[List]:
    """
    Flood fill starting from (sr, sc).
    Replaces all connected cells with the same color as (sr,sc) with new_color.
    Modifies grid in-place and returns it.
    """
    rows, cols = len(grid), len(grid[0])
    old_color = grid[sr][sc]
    if old_color == new_color:
        return grid

    stack = [(sr, sc)]
    while stack:
        r, c = stack.pop()
        if not in_bounds(r, c, rows, cols) or grid[r][c] != old_color:
            continue
        grid[r][c] = new_color
        for dr, dc in dirs:
            stack.append((r + dr, c + dc))

    return grid


# ──────────────────────────────────────────────
# 4. Island Counting (connected components on grid)
#    Count connected components of land cells ('1' or 1).
# ──────────────────────────────────────────────

def count_islands(grid: List[List[str]],
                  land: str = '1', dirs: List[Tuple] = DIRS_4) -> int:
    """
    Returns the number of islands (connected components of land cells).
    Modifies grid in-place by marking visited land as '0' (sink).
    """
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def _dfs(r: int, c: int):
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if not in_bounds(cr, cc, rows, cols) or grid[cr][cc] != land:
                continue
            grid[cr][cc] = '0'    # sink visited land
            for dr, dc in dirs:
                stack.append((cr + dr, cc + dc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == land:
                _dfs(r, c)
                count += 1

    return count


def count_islands_copy(grid: List[List[str]],
                        land: str = '1', dirs: List[Tuple] = DIRS_4) -> int:
    """Non-destructive island count using a visited set."""
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0

    def _dfs(r: int, c: int):
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited or not in_bounds(cr, cc, rows, cols) or grid[cr][cc] != land:
                continue
            visited.add((cr, cc))
            for dr, dc in dirs:
                stack.append((cr + dr, cc + dc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == land and (r, c) not in visited:
                _dfs(r, c)
                count += 1

    return count


# ──────────────────────────────────────────────
# 5. DFS on Grid — general component extraction
#    Returns all cells in the connected component containing (sr, sc).
# ──────────────────────────────────────────────

def dfs_component(grid: List[List], sr: int, sc: int,
                   passable: Callable = lambda x: x == 1,
                   dirs: List[Tuple] = DIRS_4) -> List[Tuple[int, int]]:
    """
    Returns list of all (r, c) cells reachable from (sr, sc) via passable cells.
    Does NOT modify the grid.
    """
    rows, cols = len(grid), len(grid[0])
    visited = set()
    component = []
    stack = [(sr, sc)]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited or not in_bounds(r, c, rows, cols) or not passable(grid[r][c]):
            continue
        visited.add((r, c))
        component.append((r, c))
        for dr, dc in dirs:
            stack.append((r + dr, c + dc))
    return component


# ──────────────────────────────────────────────
# 6. BFS with path reconstruction on grid
# ──────────────────────────────────────────────

def bfs_grid_path(grid: List[List], sr: int, sc: int,
                   er: int, ec: int,
                   passable: Callable = lambda x: x != '#',
                   dirs: List[Tuple] = DIRS_4) -> Optional[List[Tuple[int, int]]]:
    """
    BFS shortest path on grid; returns the path as list of (r,c) cells, or None.
    """
    rows, cols = len(grid), len(grid[0])
    if not passable(grid[sr][sc]) or not passable(grid[er][ec]):
        return None
    visited = [[False] * cols for _ in range(rows)]
    parent = [[None] * cols for _ in range(rows)]
    visited[sr][sc] = True
    queue: deque = deque([(sr, sc)])

    while queue:
        r, c = queue.popleft()
        if r == er and c == ec:
            # Reconstruct path
            path = []
            cr, cc = er, ec
            while (cr, cc) != (sr, sc):
                path.append((cr, cc))
                cr, cc = parent[cr][cc]
            path.append((sr, sc))
            path.reverse()
            return path
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and not visited[nr][nc] and passable(grid[nr][nc]):
                visited[nr][nc] = True
                parent[nr][nc] = (r, c)
                queue.append((nr, nc))
    return None


# ──────────────────────────────────────────────
# 7. 0-1 BFS on Grid (edge weights 0 or 1)
#    e.g., doors cost 1 to open, open corridors cost 0.
# ──────────────────────────────────────────────

def bfs_01_grid(grid: List[List[int]], sr: int, sc: int,
                 dirs: List[Tuple] = DIRS_4) -> List[List[int]]:
    """
    0-1 BFS on grid where grid[r][c] is the cost to enter cell (r, c).
    Returns dist[r][c] = min cost to reach (r,c) from (sr,sc).
    -1 if unreachable.
    """
    rows, cols = len(grid), len(grid[0])
    INF = float('inf')
    dist = [[INF] * cols for _ in range(rows)]
    dist[sr][sc] = 0
    dq: deque = deque([(sr, sc)])

    while dq:
        r, c = dq.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols):
                w = grid[nr][nc]
                nd = dist[r][c] + w
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    if w == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))

    return [[d if d != INF else -1 for d in row] for row in dist]


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Grid BFS shortest path
    maze = [
        ['.', '.', '#', '.'],
        ['#', '.', '#', '.'],
        ['.', '.', '.', '.'],
        ['.', '#', '#', '.'],
    ]
    dist = bfs_grid_shortest(maze, 0, 0, 3, 3, passable=lambda x: x != '#')
    print("Shortest path (0,0)->(3,3):", dist)   # some positive integer

    path = bfs_grid_path(maze, 0, 0, 3, 3, passable=lambda x: x != '#')
    print("Path:", path)

    # Multi-source BFS: distance to nearest 0
    matrix = [[0,0,0],[0,1,0],[1,1,1]]
    result = multi_source_bfs_grid(matrix,
                                   is_source=lambda x: x == 0,
                                   passable=lambda x: True)
    print("Distance to nearest 0:\n", '\n'.join(str(r) for r in result))

    # Island counting
    island_grid = [
        ['1','1','0','0','0'],
        ['1','1','0','0','0'],
        ['0','0','1','0','0'],
        ['0','0','0','1','1'],
    ]
    from copy import deepcopy
    print("Islands:", count_islands(deepcopy(island_grid)))   # 3

    # Flood fill
    ff_grid = [[1,1,1],[1,1,0],[1,0,1]]
    flood_fill(ff_grid, 1, 1, 2)
    print("After flood fill:", ff_grid)   # [[2,2,2],[2,2,0],[2,0,1]]
