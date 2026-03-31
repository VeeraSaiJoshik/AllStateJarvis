"""
NAME: Chinese Remainder Theorem (CRT)
TAGS: crt, chinese-remainder-theorem, modular-arithmetic, number-theory
DESCRIPTION: Solves systems of simultaneous congruences x ≡ r_i (mod m_i). The standard
    CRT requires pairwise coprime moduli; the generalized version handles arbitrary moduli
    using GCD-based merging. Use whenever you need to reconstruct a value from its residues
    or combine modular constraints across different moduli.
COMPLEXITY: Time O(n log M) for n congruences with max modulus M; Space O(1)
"""

from math import gcd


# ── Extended GCD helper ───────────────────────────────────────────────────────
def ext_gcd(a, b):
    """Returns (g, x, y) s.t. a*x + b*y = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = ext_gcd(b, a % b)
    return g, y, x - (a // b) * y


# ── Standard CRT (pairwise coprime moduli) ────────────────────────────────────
def crt(remainders, moduli):
    """
    Solve x ≡ r_i (mod m_i) where all m_i are pairwise coprime.
    Returns unique x in [0, M) where M = product of all moduli.

    Example: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)  -> x = 23, M = 105
    """
    M = 1
    for m in moduli:
        M *= m

    x = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        _, inv, _ = ext_gcd(Mi, m)
        x += r * Mi * inv
    return x % M

# Example:
# crt([2, 3, 2], [3, 5, 7]) -> 23  (23 mod 3=2, 23 mod 5=3, 23 mod 7=2)


# ── Generalized CRT (arbitrary moduli, pairwise merge) ────────────────────────
def crt_merge(r1, m1, r2, m2):
    """
    Merge two congruences: x ≡ r1 (mod m1) and x ≡ r2 (mod m2).
    Returns (r, m) such that x ≡ r (mod m), or None if no solution.
    Works even when gcd(m1, m2) > 1.

    Algorithm: find x = r1 + m1*t such that r1 + m1*t ≡ r2 (mod m2)
    => m1*t ≡ (r2 - r1) (mod m2) — solvable iff gcd(m1,m2) | (r2-r1)
    """
    g, p, _ = ext_gcd(m1, m2)
    if (r2 - r1) % g != 0:
        return None   # No solution

    lcm = m1 // g * m2
    diff = (r2 - r1) // g
    t = diff * p % (m2 // g)
    r = (r1 + m1 * t) % lcm
    return r, lcm

def crt_general(remainders, moduli):
    """
    Solve system of congruences with arbitrary (possibly non-coprime) moduli.
    Merges congruences one by one. Returns (x, M) or None if inconsistent.
    """
    r, m = remainders[0], moduli[0]
    for ri, mi in zip(remainders[1:], moduli[1:]):
        result = crt_merge(r, m, ri, mi)
        if result is None:
            return None  # Inconsistent system
        r, m = result
    return r % m, m

# Example:
# crt_general([0, 0], [6, 10]) -> (0, 30)   gcd(6,10)=2 divides 0-0=0
# crt_general([1, 2], [4, 6])  -> (10, 12)  x≡1(mod4), x≡2(mod6) -> x=10, M=12
# crt_general([1, 0], [2, 4])  -> None      inconsistent


# ── Garner's Algorithm (CRT for large numbers mod p) ─────────────────────────
def garner(remainders, moduli, p):
    """
    Reconstruct a large integer x from its residues mod m_i, then return x mod p.
    Useful when x could be very large (e.g., x < product of all moduli).
    moduli must be pairwise coprime.

    x = a_0 + a_1*m_0 + a_2*m_0*m_1 + ...  (mixed-radix representation)
    """
    # We'll use moduli + [p] if p is not among them
    n = len(remainders)
    # Build mixed-radix coefficients
    coeffs = list(remainders)
    prefix_mod = [1] * n   # prefix_mod[i] = m_0 * m_1 * ... * m_{i-1} mod p

    for i in range(1, n):
        # subtract earlier terms from coeffs[i]
        for j in range(i):
            coeffs[i] = (coeffs[i] - coeffs[j]) * pow(moduli[j], -1, moduli[i]) % moduli[i]

    # Now reconstruct x mod p
    result = 0
    cur_mod = 1
    for i in range(n):
        result = (result + coeffs[i] * cur_mod) % p
        cur_mod = cur_mod * moduli[i] % p
    return result

# Example:
# garner([2, 3, 2], [3, 5, 7], 10**9+7) -> 23


# ── Application: Recovering Large nCr from NTT primes ────────────────────────
# Common competition technique: compute answer mod several NTT-friendly primes,
# then use CRT + Garner to recover the true answer.
NTT_PRIMES = [998244353, 985661441, 754974721]  # Common NTT primes

def recover_from_ntt_primes(residues):
    """
    Given residues[i] = answer mod NTT_PRIMES[i], recover answer mod 10^9+7.
    Assumes the true answer fits in the product of NTT_PRIMES.
    """
    return garner(residues, NTT_PRIMES, 10**9 + 7)


# ── CRT for Two Moduli (Inline version for speed) ─────────────────────────────
def crt2(r1, m1, r2, m2):
    """
    Fast inline CRT for exactly 2 congruences (coprime moduli).
    Returns x in [0, m1*m2) such that x ≡ r1 (mod m1), x ≡ r2 (mod m2).
    """
    # x = r1 + m1 * t, need m1*t ≡ (r2-r1) (mod m2)
    # t = (r2 - r1) * inv(m1) mod m2
    inv_m1_m2 = pow(m1, -1, m2)
    t = (r2 - r1) % m2 * inv_m1_m2 % m2
    return r1 + m1 * t

# Example:
# crt2(2, 3, 3, 5) -> 8  (8 mod 3=2, 8 mod 5=3)
# But crt([2,3],[3,5]) -> 8 too
