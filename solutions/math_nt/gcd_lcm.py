"""
NAME: GCD, LCM, and Extended Euclidean Algorithm
TAGS: gcd, lcm, extended-euclidean, bezout, number-theory
DESCRIPTION: Computes GCD and LCM of integers, and uses the Extended Euclidean Algorithm
    to find Bezout coefficients (x, y) such that ax + by = gcd(a, b). Essential for
    modular inverses, solving linear Diophantine equations, and the Chinese Remainder Theorem.
COMPLEXITY: Time O(log(min(a,b))); Space O(log(min(a,b))) recursive, O(1) iterative
"""

from math import gcd  # Built-in, faster than manual for simple use

# ── GCD ───────────────────────────────────────────────────────────────────────
def my_gcd(a, b):
    """Euclidean GCD (reference; prefer math.gcd in competitions)."""
    while b:
        a, b = b, a % b
    return a

def gcd_multi(*args):
    """GCD of multiple numbers."""
    result = args[0]
    for x in args[1:]:
        result = gcd(result, x)
    return result

# Example:
# gcd(12, 18)          -> 6
# gcd_multi(12, 18, 24) -> 6


# ── LCM ───────────────────────────────────────────────────────────────────────
def lcm(a, b):
    """LCM of two numbers. Divide before multiply to prevent overflow."""
    return a // gcd(a, b) * b

def lcm_multi(*args):
    """LCM of multiple numbers."""
    result = args[0]
    for x in args[1:]:
        result = lcm(result, x)
    return result

# Example:
# lcm(4, 6)             -> 12
# lcm_multi(4, 6, 10)   -> 60
# Python 3.9+: math.lcm(a, b) or math.lcm(*args) is built-in


# ── Extended Euclidean Algorithm ──────────────────────────────────────────────
def extended_gcd(a, b):
    """
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    Bezout's identity: gcd(a, b) = a*x + b*y always has a solution.
    """
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

def extended_gcd_iter(a, b):
    """Iterative version (no recursion limit issues)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    # old_r = gcd, old_s = x, old_t = y
    return old_r, old_s, old_t

# Example:
# extended_gcd(35, 15) -> (5, 1, -2)  because 35*1 + 15*(-2) = 5
# extended_gcd(3, 7)   -> (1, -2, 1)  because 3*(-2) + 7*1 = 1


# ── Linear Diophantine Equation ───────────────────────────────────────────────
def solve_diophantine(a, b, c):
    """
    Find integer solution (x0, y0) to ax + by = c.
    Returns None if no solution exists (c % gcd(a,b) != 0).
    General solution: x = x0 + (b/g)*t,  y = y0 - (a/g)*t  for any integer t.
    """
    g, x, y = extended_gcd(abs(a), abs(b))
    if c % g != 0:
        return None  # No integer solution
    x *= c // g
    y *= c // g
    if a < 0: x = -x
    if b < 0: y = -y
    return x, y, g  # One particular solution + gcd

# Example:
# solve_diophantine(3, 5, 1)  -> x0, y0 such that 3*x0 + 5*y0 = 1
# solve_diophantine(6, 4, 3)  -> None  (gcd(6,4)=2 does not divide 3)


# ── Modular Inverse via Extended GCD ──────────────────────────────────────────
def mod_inverse(a, m):
    """
    Modular inverse of a mod m using extended GCD.
    Requires gcd(a, m) = 1. Returns x such that a*x ≡ 1 (mod m).
    Prefer pow(a, -1, m) in Python 3.8+ for simplicity.
    """
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Inverse doesn't exist: gcd({a}, {m}) = {g}")
    return x % m

# Example:
# mod_inverse(3, 7)   -> 5   (3*5 = 15 ≡ 1 mod 7)
# pow(3, -1, 7)       -> 5   (Python 3.8+ built-in)


# ── GCD of Array / GCD-related Tricks ────────────────────────────────────────
def array_gcd(arr):
    """GCD of entire array. Useful for divisibility checks."""
    from functools import reduce
    return reduce(gcd, arr)

def count_pairs_gcd_k(arr, k):
    """Count pairs (i,j) where gcd(arr[i], arr[j]) == k."""
    from collections import Counter
    count = Counter(x for x in arr if x % k == 0)
    # Normalize by dividing by k
    normalized = Counter(x // k for x in arr if x % k == 0)
    # Use inclusion-exclusion with Euler's totient / Mobius
    # For simple version: iterate over multiples
    freq = [0] * (max(normalized) + 2) if normalized else [0]
    for v, c in normalized.items():
        freq[v] = c
    result_at_least = [0] * len(freq)
    for d in range(1, len(freq)):
        for mult in range(d, len(freq), d):
            result_at_least[d] += freq[mult]
    # pairs with gcd divisible by d*k: C(result_at_least[d], 2)
    # Use Mobius inversion for exact count
    pairs_ge = [x * (x - 1) // 2 for x in result_at_least]
    exact = [0] * len(freq)
    for d in range(len(freq) - 1, 0, -1):
        exact[d] = pairs_ge[d]
        for mult in range(2 * d, len(freq), d):
            exact[d] -= exact[mult]
    return exact[1] if len(exact) > 1 else 0

# Example:
# array_gcd([12, 18, 24]) -> 6
