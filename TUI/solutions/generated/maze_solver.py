"""
NAME: Maze Solver - Multiple Algorithms
TAGS: maze, bfs, dfs, dijkstra, a-star, pathfinding, grid-traversal, graphs
DESCRIPTION: Solve maze problems using BFS (shortest path), DFS (any path), Dijkstra (weighted), and A* (optimal heuristic). Supports configurable cell markers and multiple path-finding strategies.
COMPLEXITY: Time: O(m*n), Space: O(m*n)
CODE:
"""

from collections import deque
from typing import List, Tuple, Optional, Set
import heapq

# ============================================================================
# CONFIGURABLE MARKERS - Change these to match problem requirements
# ============================================================================

EDGE_MARKER = 'X'       # Outer boundary/wall
WALL_MARKER = 'x'       # Interior walls
START_MARKER = 'S'      # Starting position
FINISH_MARKER = 'F'     # Goal/finish position
EMPTY_MARKER = ' '      # Walkable empty space
PATH_MARKER = '*'       # Path visualization (for output)

# Alternative common marker sets:
# - '#' for walls, '.' for empty, 'S'/'E' for start/end
# - '1' for walls, '0' for empty, '2'/'3' for start/end
# - 'W' for walls, 'O' for open, 'A'/'B' for start/end


# ============================================================================
# CORE MAZE SOLVER CLASS
# ============================================================================

class MazeSolver:
    """
    Comprehensive maze solver supporting multiple algorithms.
    """

    def __init__(self, maze: List[List[str]]):
        """
        Initialize maze solver.

        Args:
            maze: 2D grid with characters defined by marker constants
        """
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if maze else 0
        self.start = self._find_position(START_MARKER)
        self.finish = self._find_position(FINISH_MARKER)

    def _find_position(self, marker: str) -> Optional[Tuple[int, int]]:
        """Find position of a marker in the maze."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == marker:
                    return (i, j)
        return None

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if position is valid and walkable."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        cell = self.maze[row][col]
        # Can walk on: empty, start, or finish
        return cell not in [EDGE_MARKER, WALL_MARKER]

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-directional)."""
        row, col = pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        neighbors = []

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid(new_row, new_col):
                neighbors.append((new_row, new_col))

        return neighbors

    def _reconstruct_path(self, came_from: dict, start: Tuple[int, int],
                         end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary."""
        path = []
        current = end

        while current != start:
            path.append(current)
            if current not in came_from:
                return []  # No path found
            current = came_from[current]

        path.append(start)
        path.reverse()
        return path


# ============================================================================
# BFS - Shortest Path (Unweighted)
# ============================================================================

    def solve_bfs(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Solve maze using Breadth-First Search.
        Guarantees shortest path in unweighted maze.

        Returns:
            (found, path, steps_explored)
            - found: True if path exists
            - path: List of (row, col) positions from start to finish
            - steps_explored: Number of cells explored

        Time: O(m*n), Space: O(m*n)
        """
        if not self.start or not self.finish:
            return (False, [], 0)

        queue = deque([self.start])
        visited = {self.start}
        came_from = {}
        steps = 0

        while queue:
            current = queue.popleft()
            steps += 1

            if current == self.finish:
                path = self._reconstruct_path(came_from, self.start, self.finish)
                return (True, path, steps)

            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

        return (False, [], steps)


# ============================================================================
# DFS - Any Path (May not be shortest)
# ============================================================================

    def solve_dfs(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Solve maze using Depth-First Search.
        Finds a path but NOT guaranteed to be shortest.

        Returns:
            (found, path, steps_explored)

        Time: O(m*n), Space: O(m*n)
        """
        if not self.start or not self.finish:
            return (False, [], 0)

        stack = [self.start]
        visited = {self.start}
        came_from = {}
        steps = 0

        while stack:
            current = stack.pop()
            steps += 1

            if current == self.finish:
                path = self._reconstruct_path(came_from, self.start, self.finish)
                return (True, path, steps)

            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append(neighbor)

        return (False, [], steps)


# ============================================================================
# A* - Optimal with Heuristic
# ============================================================================

    def solve_astar(self) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Solve maze using A* search algorithm.
        Uses Manhattan distance heuristic for grid.
        Optimal and typically faster than BFS.

        Returns:
            (found, path, steps_explored)

        Time: O(m*n*log(m*n)), Space: O(m*n)
        """
        if not self.start or not self.finish:
            return (False, [], 0)

        def heuristic(pos: Tuple[int, int]) -> int:
            """Manhattan distance to goal."""
            return abs(pos[0] - self.finish[0]) + abs(pos[1] - self.finish[1])

        # Priority queue: (f_score, counter, position)
        # f_score = g_score + heuristic
        counter = 0
        heap = [(heuristic(self.start), counter, self.start)]
        came_from = {}
        g_score = {self.start: 0}
        steps = 0

        while heap:
            _, _, current = heapq.heappop(heap)
            steps += 1

            if current == self.finish:
                path = self._reconstruct_path(came_from, self.start, self.finish)
                return (True, path, steps)

            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    counter += 1
                    heapq.heappush(heap, (f_score, counter, neighbor))

        return (False, [], steps)


# ============================================================================
# DIJKSTRA - Weighted Shortest Path
# ============================================================================

    def solve_dijkstra(self, weights: Optional[List[List[int]]] = None) -> \
            Tuple[bool, List[Tuple[int, int]], int]:
        """
        Solve maze using Dijkstra's algorithm.
        Supports weighted edges (useful for cost-based mazes).

        Args:
            weights: Optional 2D array of weights. If None, all weights = 1.

        Returns:
            (found, path, steps_explored)

        Time: O(m*n*log(m*n)), Space: O(m*n)
        """
        if not self.start or not self.finish:
            return (False, [], 0)

        # Priority queue: (distance, counter, position)
        counter = 0
        heap = [(0, counter, self.start)]
        distances = {self.start: 0}
        came_from = {}
        steps = 0

        while heap:
            current_dist, _, current = heapq.heappop(heap)
            steps += 1

            if current == self.finish:
                path = self._reconstruct_path(came_from, self.start, self.finish)
                return (True, path, steps)

            if current_dist > distances.get(current, float('inf')):
                continue

            for neighbor in self._get_neighbors(current):
                # Get edge weight
                if weights:
                    weight = weights[neighbor[0]][neighbor[1]]
                else:
                    weight = 1

                new_dist = current_dist + weight

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(heap, (new_dist, counter, neighbor))

        return (False, [], steps)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_maze(maze: List[List[str]], path: List[Tuple[int, int]] = None) -> str:
    """
    Create visual representation of maze with optional path overlay.

    Args:
        maze: 2D grid
        path: Optional list of (row, col) positions to mark

    Returns:
        String representation of maze
    """
    path_set = set(path) if path else set()
    result = []

    for i, row in enumerate(maze):
        line = ""
        for j, cell in enumerate(row):
            if (i, j) in path_set and cell not in [START_MARKER, FINISH_MARKER]:
                line += PATH_MARKER
            else:
                line += cell
        result.append(line)

    return "\n".join(result)


def parse_maze(maze_str: str) -> List[List[str]]:
    """
    Parse maze from string representation.

    Args:
        maze_str: Multi-line string with maze layout

    Returns:
        2D list of characters
    """
    lines = maze_str.strip().split('\n')
    return [list(line) for line in lines]


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

def test_maze_solver():
    """Comprehensive test cases."""

    # Test Case 1: Simple maze
    maze1_str = """
XXXXXXXXXXXXX
X           X
X XXX XXXXX X
X X   X   X X
X X XXX X X X
X X     X   X
X XXXXXXX XXX
XS          X
XXXXXXXXXXXXX
    """.strip()

    maze1 = parse_maze(maze1_str)
    # Add finish marker
    maze1[7][11] = FINISH_MARKER

    print("="*60)
    print("TEST CASE 1: Simple Maze")
    print("="*60)
    print("\nOriginal Maze:")
    print(visualize_maze(maze1))

    solver = MazeSolver(maze1)

    # Test all algorithms
    algorithms = [
        ("BFS (Shortest Path)", solver.solve_bfs),
        ("DFS (Any Path)", solver.solve_dfs),
        ("A* (Optimal)", solver.solve_astar),
        ("Dijkstra", solver.solve_dijkstra),
    ]

    for name, method in algorithms:
        found, path, steps = method()
        print(f"\n{name}:")
        print(f"  Found: {found}")
        print(f"  Path length: {len(path) if found else 0}")
        print(f"  Steps explored: {steps}")

        if found and name == "BFS (Shortest Path)":
            print("\nPath visualization:")
            print(visualize_maze(maze1, path))

    # Test Case 2: No solution
    maze2_str = """
XXXXXXXXX
XS X    X
X  X    X
X  X    X
X  X   FX
XXXXXXXXX
    """.strip()

    maze2 = parse_maze(maze2_str)

    print("\n" + "="*60)
    print("TEST CASE 2: No Solution Maze")
    print("="*60)
    print("\nMaze:")
    print(visualize_maze(maze2))

    solver2 = MazeSolver(maze2)
    found, path, steps = solver2.solve_bfs()
    print(f"\nBFS Result:")
    print(f"  Found: {found}")
    print(f"  Path length: {len(path)}")
    print(f"  Steps explored: {steps}")

    # Test Case 3: Complex maze
    maze3_str = """
XXXXXXXXXXXXXXXXXXXXXXX
XS                    X
X XXXXX XXXXX XXXXX X X
X X   X X   X X   X X X
X X X X X X X X X X X X
X X X X X X X X X X X X
X X X   X X   X X   X X
X X XXXXX XXXXX XXXXX X
X X                   X
X XXXXXXXXXXXXXXXXXXX X
X                    FX
XXXXXXXXXXXXXXXXXXXXXXX
    """.strip()

    maze3 = parse_maze(maze3_str)

    print("\n" + "="*60)
    print("TEST CASE 3: Complex Maze - Algorithm Comparison")
    print("="*60)
    print("\nMaze:")
    print(visualize_maze(maze3))

    solver3 = MazeSolver(maze3)

    print("\nAlgorithm Performance Comparison:")
    print("-" * 60)
    print(f"{'Algorithm':<20} {'Found':<8} {'Path Len':<10} {'Explored':<10}")
    print("-" * 60)

    for name, method in algorithms:
        found, path, steps = method()
        print(f"{name:<20} {str(found):<8} {len(path):<10} {steps:<10}")


def test_custom_markers():
    """Test with different marker configurations."""

    print("\n" + "="*60)
    print("TEST CASE 4: Custom Markers (# walls, . empty, S/E start/end)")
    print("="*60)

    # Temporarily change markers
    global EDGE_MARKER, WALL_MARKER, START_MARKER, FINISH_MARKER, EMPTY_MARKER

    old_markers = (EDGE_MARKER, WALL_MARKER, START_MARKER, FINISH_MARKER, EMPTY_MARKER)

    EDGE_MARKER = '#'
    WALL_MARKER = '#'
    START_MARKER = 'S'
    FINISH_MARKER = 'E'
    EMPTY_MARKER = '.'

    maze_str = """
###########
#S........#
#.###.###.#
#.#...#...#
#.#.###.###
#.........E
###########
    """.strip()

    maze = parse_maze(maze_str)
    print("\nMaze:")
    print(visualize_maze(maze))

    solver = MazeSolver(maze)
    found, path, steps = solver.solve_astar()

    print(f"\nA* Result:")
    print(f"  Found: {found}")
    print(f"  Path length: {len(path)}")

    if found:
        print("\nPath visualization:")
        print(visualize_maze(maze, path))

    # Restore original markers
    EDGE_MARKER, WALL_MARKER, START_MARKER, FINISH_MARKER, EMPTY_MARKER = old_markers


if __name__ == "__main__":
    test_maze_solver()
    test_custom_markers()

    print("\n" + "="*60)
    print("USAGE EXAMPLE")
    print("="*60)
    print("""
# Create a maze (use configurable markers at top of file)
maze = [
    ['X', 'X', 'X', 'X', 'X'],
    ['X', 'S', ' ', ' ', 'X'],
    ['X', 'x', 'x', ' ', 'X'],
    ['X', ' ', ' ', 'F', 'X'],
    ['X', 'X', 'X', 'X', 'X'],
]

# Solve it
solver = MazeSolver(maze)

# Use BFS for shortest path
found, path, steps = solver.solve_bfs()

# Or use A* for optimal performance
found, path, steps = solver.solve_astar()

# Visualize the solution
if found:
    print(visualize_maze(maze, path))
    print(f"Path length: {len(path)}")
    """)
