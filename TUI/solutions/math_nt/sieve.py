"""
NAME: Sieve of Eratosthenes (Standard, Linear, Segmented, SPF)
TAGS: sieve, primes, number-theory, preprocessing
DESCRIPTION: Generates all primes up to N efficiently. Use standard sieve for O(N log log N)
    prime generation, linear sieve for O(N) with smallest prime factors, and segmented sieve
    when N is large but the range [L, R] is manageable. SPF enables O(log N) factorization.
COMPLEXITY: Time O(N log log N) standard, O(N) linear; Space O(N)
"""

# ── Standard Sieve of Eratosthenes ───────────────────────────────────────────
def sieve(n):
    """Return list of primes up to n (inclusive)."""
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

# Example: sieve(30) -> [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


# ── Smallest Prime Factor (SPF) Sieve ────────────────────────────────────────
def spf_sieve(n):
    """
    Returns spf[i] = smallest prime factor of i, for 2 <= i <= n.
    Enables O(log n) factorization of any number up to n.
    """
    spf = list(range(n + 1))
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:          # i is prime
            for j in range(i*i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def factorize_with_spf(x, spf):
    """Factorize x in O(log x) using precomputed SPF table."""
    factors = {}
    while x > 1:
        p = spf[x]
        while x % p == 0:
            factors[p] = factors.get(p, 0) + 1
            x //= p
    return factors

# Example:
# spf = spf_sieve(20)  -> spf[12] = 2, spf[15] = 3
# factorize_with_spf(12, spf) -> {2: 2, 3: 1}


# ── Linear Sieve ─────────────────────────────────────────────────────────────
def linear_sieve(n):
    """
    O(N) sieve. Returns (is_prime array, primes list, spf array).
    Each composite is crossed out exactly once by its smallest prime factor.
    """
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = is_prime[1] = 0
    primes = []
    spf = [0] * (n + 1)

    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = 0
            spf[i * p] = p
            if i % p == 0:
                break   # key line: ensures each composite hit exactly once
    return is_prime, primes, spf

# Example:
# _, primes, spf = linear_sieve(30)
# primes -> [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


# ── Segmented Sieve ───────────────────────────────────────────────────────────
def segmented_sieve(lo, hi):
    """
    Returns list of primes in [lo, hi].
    Uses O(sqrt(hi)) memory instead of O(hi). Good when hi is up to ~1e12.
    """
    import math
    limit = int(math.isqrt(hi)) + 1
    small_primes = sieve(limit)

    is_prime_seg = bytearray([1]) * (hi - lo + 1)
    if lo == 1:
        is_prime_seg[0] = 0
    if lo == 0:
        is_prime_seg[0] = is_prime_seg[1] = 0

    for p in small_primes:
        # first multiple of p in [lo, hi]
        start = max(p * p, ((lo + p - 1) // p) * p)
        if start == p:
            start += p
        for j in range(start, hi + 1, p):
            is_prime_seg[j - lo] = 0

    return [lo + i for i in range(hi - lo + 1) if is_prime_seg[i]]

# Example:
# segmented_sieve(10, 30) -> [11, 13, 17, 19, 23, 29]
# segmented_sieve(10**11, 10**11 + 1000) -> primes in that range


# ── Prime Counting / Utility ──────────────────────────────────────────────────
def is_prime_miller(n):
    """Deterministic Miller-Rabin for n < 3.3 * 10^24. Fast primality check."""
    if n < 2: return False
    if n == 2 or n == 3 or n == 5 or n == 7: return True
    if n % 2 == 0: return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Deterministic witnesses for n < 3,317,044,064,679,887,385,961,981
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n: continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            return False
    return True

# Example:
# is_prime_miller(10**18 + 9) -> True/False  (works for large primes)
