"""
NAME: Hexadecimal Addition Carry Count (Arkansas All-State 2019 Problem 4)
TAGS: number-theory, base-conversion, simulation, math
DESCRIPTION: Count carry operations when adding two hexadecimal numbers digit by digit.
             In hex addition, if digit sum >= 16, a carry is generated.
COMPLEXITY: Time O(n), Space O(n)
"""

def solve():
    """Process test cases to count carry operations in hex addition."""
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            continue

        hex1, hex2 = parts[0].upper(), parts[1].upper()

        # Pad to same length
        max_len = max(len(hex1), len(hex2))
        hex1 = hex1.zfill(max_len)
        hex2 = hex2.zfill(max_len)

        carry = 0
        carry_count = 0

        # Process from right to left
        for i in range(max_len - 1, -1, -1):
            digit1 = int(hex1[i], 16)
            digit2 = int(hex2[i], 16)

            total = digit1 + digit2 + carry

            if total >= 16:
                carry = 1
                carry_count += 1
            else:
                carry = 0

        if carry_count == 0:
            print("No carry operation.")
        elif carry_count == 1:
            print("1 carry operation.")
        else:
            print(f"{carry_count} carry operations.")

        print()  # Blank line after output


if __name__ == "__main__":
    solve()


# ─── Test Cases ────────────────────────────────────────────────────────────────

"""
Example Input:
123 456

ABC 754A

ABC 743

Expected Output:
No carry operation.

3 carry operations.

1 carry operation.
"""
