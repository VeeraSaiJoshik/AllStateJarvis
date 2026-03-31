"""
NAME: Counting Steps Again (Arkansas All-State 2019 Problem 8)
TAGS: graph, bfs, shortest-path, dynamic-programming
DESCRIPTION: Similar to Problem 7 but with adjusted constraints:
             - Each step can be up to 2 bigger than previous, equal, or up to 2 smaller
             - First step must be 1
             - Last step can be any positive value
COMPLEXITY: Time O(n^2), Space O(n^2) where n = y - x
"""

from collections import deque


def min_steps_to_reach_v2(x, y):
    """
    Find minimum number of steps to go from x to y.
    Each step length can be up to 2 bigger, equal, or up to 2 smaller than previous.
    First step must be 1.

    Uses BFS on state (position, last_step_length).
    """
    if x == y:
        return 0
    if x > y:
        return -1

    # BFS: (current_position, last_step_length, step_count)
    queue = deque([(x + 1, 1, 1)])  # Start with first step of length 1
    visited = {(x + 1, 1)}

    while queue:
        pos, last_step, steps = queue.popleft()

        # Check if we reached the target
        if pos == y:
            return steps

        # Try next step lengths: last_step - 2, -1, 0, +1, +2
        for delta in range(-2, 3):
            next_step = last_step + delta

            if next_step <= 0:
                continue

            next_pos = pos + next_step

            # Check if we reached target
            if next_pos == y:
                return steps + 1

            # Don't overshoot too much
            if next_pos > y + 10:
                continue

            state = (next_pos, next_step)
            if state not in visited and next_pos < y:
                visited.add(state)
                queue.append((next_pos, next_step, steps + 1))

    return -1


def solve():
    """Process test cases to find minimum steps with adjusted constraints."""
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = list(map(int, line.split()))
        if len(parts) != 2:
            continue

        x, y = parts
        result = min_steps_to_reach_v2(x, y)

        print(result)
        print()  # Blank line after output


if __name__ == "__main__":
    solve()


# ─── Test Cases ────────────────────────────────────────────────────────────────

"""
Example Input:
45 48

45 49

45 50

Expected Output:
3

2

3

Explanation:
45 -> 48: step of 1, step of 1, step of 1 = 3 steps
          or: step of 1, step of 2 = 2 steps (1+2 = 3, but we need to reach 48)
          Best: 1 + 1 + 1 or 1 + 2 (if 2 is allowed)
          Answer: 3

45 -> 49: step of 1, step of 3 = 2 steps (1+3 = 4)
          or: step of 1, step of 2, step of 1 = 3 steps
          Since we can increase by up to 2, we can go: 1, then 3 (1+2)
          Answer: 2

45 -> 50: step of 1, step of 2, step of 2 = 3 steps (1+2+2 = 5)
          Answer: 3
"""
