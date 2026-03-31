"""
NAME: Analyze and Display (Arkansas All-State 2019 Problem 2)
TAGS: array, sorting, conditional-logic, ad-hoc
DESCRIPTION: Given integers, output them based on their sign pattern.
             - All positive: increasing order
             - All negative: decreasing order
             - Mixed positive and negative: alternate negative then positive, both non-decreasing
             - All zero or unspecified: "The problem does not specify how to output!"
COMPLEXITY: Time O(n log n), Space O(n)
"""

def solve():
    """Process test cases and output sorted numbers based on sign patterns."""
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        n = int(line)
        numbers_line = sys.stdin.readline().strip()
        numbers = list(map(int, numbers_line.split()))

        positives = sorted([x for x in numbers if x > 0])
        negatives = sorted([x for x in numbers if x < 0])
        has_zero = 0 in numbers

        if positives and not negatives:
            # All positive: increasing order
            print(' '.join(map(str, positives)))
        elif negatives and not positives:
            # All negative: decreasing order (reverse)
            print(' '.join(map(str, reversed(negatives))))
        elif positives and negatives:
            # Mixed: alternate negative and positive, both increasing
            result = []
            i, j = 0, 0
            while i < len(negatives) or j < len(positives):
                if i < len(negatives):
                    result.append(negatives[i])
                    i += 1
                if j < len(positives):
                    result.append(positives[j])
                    j += 1
            print(' '.join(map(str, result)))
        else:
            # All zeros or unspecified case
            print("The problem does not specify how to output!")

        print()  # Blank line after output


if __name__ == "__main__":
    solve()


# ─── Test Cases ────────────────────────────────────────────────────────────────

"""
Example Input:
5
4 1 3 8 7

5
-4 -1 -3 -8 -7

6
-4 1 -3 8 7 2

5
-4 1 -3 8 0

Expected Output:
1 3 4 7 8

-1 -3 -4 -7 -8

-4 1 -3 2 7 8

The problem does not specify how to output!
"""
