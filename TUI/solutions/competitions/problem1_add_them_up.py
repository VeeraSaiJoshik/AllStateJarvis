"""
NAME: Add Them Up (Arkansas All-State 2019 Problem 1)
TAGS: input-parsing, array, simulation, ad-hoc
DESCRIPTION: Sum positive and negative integers separately from input.
             Read n integers and output the sum of positive integers and sum of negative integers.
COMPLEXITY: Time O(n), Space O(1)
"""

def solve():
    """Read test cases and output sums of positive and negative integers."""
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        n = int(line)
        numbers_line = sys.stdin.readline().strip()
        numbers = list(map(int, numbers_line.split()))

        positive_sum = sum(x for x in numbers if x > 0)
        negative_sum = sum(x for x in numbers if x < 0)

        print(f"The positive sum of the input is {positive_sum} and the negative sum is {negative_sum}.")
        print()  # Blank line after output


if __name__ == "__main__":
    solve()


# ─── Test Cases ────────────────────────────────────────────────────────────────

"""
Example Input:
2
6 9

5
-1 -2 -3 4 5

Expected Output:
The positive sum of the input is 15 and the negative sum is 0.

The positive sum of the input is 9 and the negative sum is -6.
"""
