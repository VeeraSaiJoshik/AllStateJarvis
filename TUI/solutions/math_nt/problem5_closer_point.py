"""
NAME: Finding Closer Point (Arkansas All-State 2019 Problem 5)
TAGS: geometry, distance, euclidean-distance, math
DESCRIPTION: Given three points, determine which of the second and third points is closer
             to the first point using Euclidean distance. If equal distance, output both.
COMPLEXITY: Time O(1), Space O(1)
"""

import math


def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def solve():
    """Process test cases to find closer points."""
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        coords = list(map(float, line.split()))
        if len(coords) != 6:
            continue

        x1, y1 = coords[0], coords[1]  # First point
        x2, y2 = coords[2], coords[3]  # Second point
        x3, y3 = coords[4], coords[5]  # Third point

        dist2 = euclidean_distance(x1, y1, x2, y2)
        dist3 = euclidean_distance(x1, y1, x3, y3)

        epsilon = 1e-9  # For floating point comparison

        if abs(dist2 - dist3) < epsilon:
            # Equal distances
            print(f"{x2} {y2} {x3} {y3}")
        elif dist2 < dist3:
            # Second point is closer
            print(f"{x2} {y2}")
        else:
            # Third point is closer
            print(f"{x3} {y3}")

        print()  # Blank line after output


if __name__ == "__main__":
    solve()


# ─── Test Cases ────────────────────────────────────────────────────────────────

"""
Example Input:
0 0 1 1 2 2

0 0 1 2 2 1

1.2 1.1 0 0 2.2 1.1

Expected Output:
1 1

1 2 2 1

2.2 1.1
"""
