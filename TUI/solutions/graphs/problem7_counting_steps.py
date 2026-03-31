"""
NAME: Counting Steps (Arkansas All-State 2019 Problem 7)
TAGS: graph, bfs, shortest-path, dynamic-programming
DESCRIPTION: Find minimum steps to get from x to y where each step can be non-negative,
             equal to, or one smaller than the previous step. The first and last steps must be 1.
             This is a BFS/DP problem on an implicit graph.
COMPLEXITY: Time O(n^2), Space O(n^2) where n = y - x
"""

from collections import deque


def min_steps_to_reach(x, y):
    """
    Find minimum number of steps to go from x to y.
    Each step length must be non-negative, equal to, or one smaller than previous step.
    First and last steps must be 1.

    Uses BFS on state (position, last_step_length).
    """
    if x == y:
        return 0
    if x > y:
        return -1  # Cannot reach if x > y with positive steps

    # BFS: (current_position, last_step_length, step_count)
    queue = deque([(x + 1, 1, 1)])  # Start with first step of length 1
    visited = {(x + 1, 1)}

    while queue:
        pos, last_step, steps = queue.popleft()

        # If we reach y, the last step must be 1
        if pos == y and last_step == 1:
            return steps

        # Try next step lengths: last_step - 1, last_step, last_step + 1
        for next_step in [last_step - 1, last_step, last_step + 1]:
            if next_step <= 0:
                continue

            next_pos = pos + next_step

            # Don't overshoot too much
            if next_pos > y + 1:
                continue

            # If we reach y, ensure last step is 1
            if next_pos == y and next_step == 1:
                return steps + 1

            state = (next_pos, next_step)
            if state not in visited and next_pos < y:
                visited.add(state)
                queue.append((next_pos, next_step, steps + 1))

    return -1  # Should not reach here for valid inputs


def solve():
    """Process test cases to find minimum steps."""
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = list(map(int, line.split()))
        if len(parts) != 2:
            continue

        x, y = parts
        result = min_steps_to_reach(x, y)

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

3

4

Explanation:
45 -> 48: step of 1, then 1, then 1 = 3 steps (1+1+1 = 3)
          or: step of 1, then 2 (invalid, last step must be 1)
          Answer: 3

45 -> 49: step of 1, step of 2, step of 1 = 3 steps (1+2+1 = 4)
          Answer: 3

45 -> 50: step of 1, step of 2, step of 1, step of 1 = 4 steps
          or: step of 1, step of 2, step of 2 (invalid, last step must be 1)
          Answer: 4
"""
