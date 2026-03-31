"""
NAME: Extract Hexadecimal Digits (Arkansas All-State 2019 Problem 3)
TAGS: string, parsing, validation, hexadecimal, ad-hoc
DESCRIPTION: Validate hexadecimal strings and extract digits with '+' separators.
             Valid hex digits: 0-9, A-F (case insensitive). Output with '+' between each digit.
COMPLEXITY: Time O(n), Space O(n)
"""

def solve():
    """Process test cases to validate and format hexadecimal strings."""
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # Check if all characters are valid hexadecimal digits
        valid_hex = all(c in '0123456789ABCDEFabcdef' for c in line)

        if valid_hex:
            # Extract and format with '+' separator
            formatted = '+'.join(line.upper())
            print(formatted)
        else:
            print("This is not a hexadecimal number.")

        print()  # Blank line after output


if __name__ == "__main__":
    solve()


# ─── Test Cases ────────────────────────────────────────────────────────────────

"""
Example Input:
9876543210ABCDEF

EFG

Expected Output:
9+8+7+6+5+4+3+2+1+0+A+B+C+D+E+F

This is not a hexadecimal number.
"""
