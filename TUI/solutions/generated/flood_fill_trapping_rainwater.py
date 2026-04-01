"""
NAME: Flood Fill - Trapping Rain Water
TAGS: flood-fill, heap, priority-queue, bfs, matrix, water-trapping, greedy
DESCRIPTION: Use flood fill with priority queue to calculate water trapped in a height map. Works for both 1D and 2D elevation maps by processing cells from lowest to highest boundary.
COMPLEXITY: Time: O(m*n*log(m*n)), Space: O(m*n)
CODE:
"""

import heapq
from typing import List

def trap_1d(height: List[int]) -> int:
    """
    Calculate water trapped in 1D elevation map.

    Two-pointer approach: water level at each position is determined by
    min(max_left, max_right) - height[i]

    Time: O(n), Space: O(1)

    Example:
        height = [0,1,0,2,1,0,1,3,2,1,2,1]
        Returns: 6

        Visual:
               █
           █▓▓▓█▓█
         █▓█▓█▓███▓█
        [0,1,0,2,1,0,1,3,2,1,2,1]
        (▓ represents trapped water)
    """
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            # Process left side
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            # Process right side
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water