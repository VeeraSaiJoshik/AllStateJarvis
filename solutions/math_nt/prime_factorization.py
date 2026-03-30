"""
NAME: Prime Factorization (Trial Division, Pollard's Rho, Divisor Functions)
TAGS: prime-factorization, pollard-rho, divisors, number-theory
DESCRIPTION: Factorizes integers efficiently using trial division for small numbers and
    Pollard's rho algorithm for large semiprimes (up to ~10^18). Includes divisor counting,
    sum of divisors, and enumeration. Use Pollard's rho when N > 10^12 and trial division otherwise.
COMPLEXITY: Trial division O(sqrt(N)); Pollard's rho O(N^(1/4) log N); Space O(log N)
"""

from math import isqrt, gcd
from random import randint


# ── Trial Division Factorization ──────────────────────────────────────────────
def factorize(n):
    """
    Factorize n using trial division. Returns dict {prime: exponent}.
    Efficient for n up to ~10^12. For larger, use pollard_rho_factorize.
    """
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Example:
# factorize(360) -> {2: 3, 3: 2, 5: 1}   (360 = 2^3 * 3^2 * 5)
# factorize(13)  -> {13: 1}


# ── Miller-Rabin Primality Test (Deterministic) ───────────────────────────────
def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3 * 10^24."""
    if n < 2: return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n in small: return True
    if any(n % p == 0 for p in small): return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1; d //= 2
    for a in small:
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            return False
    return True


# ── Pollard's Rho ─────────────────────────────────────────────────────────────
def _pollard_rho(n):
    """Find a non-trivial factor of n using Pollard's rho (Brent's variant)."""
    if n % 2 == 0: return 2
    x = randint(2, n - 1)
    y = x
    c = randint(1, n - 1)
    d = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = gcd(abs(x - y), n)
    return d if d != n else None

def pollard_rho_factorize(n):
    """
    Factorize n using Pollard's rho + Miller-Rabin. Handles n up to 10^18.
    Returns dict {prime: exponent}.
    """
    if n <= 1: return {}
    factors = {}

    def _factor(n):
        if n == 1: return
        if is_prime(n):
            factors[n] = factors.get(n, 0) + 1
            return
        # Try to find a factor
        d = n
        while d == n:
            d = _pollard_rho(n)
            if d is None: d = n  # retry
        _factor(d)
        _factor(n // d)

    _factor(n)
    return factors

# Example:
# pollard_rho_factorize(600851475143)          -> {71: 1, 839: 1, 1471: 1, 6857: 1}
# pollard_rho_factorize(10**18 + 7)            -> factorization of large semiprimes


# ── Divisor Functions ─────────────────────────────────────────────────────────
def num_divisors(factors):
    """
    Number of divisors d(n) = product of (e_i + 1) for each prime power p^e in factorization.
    Input: factors dict {p: e}
    """
    result = 1
    for e in factors.values():
        result *= (e + 1)
    return result

def sum_divisors(factors):
    """
    Sum of divisors sigma(n) = product of (p^(e+1) - 1) / (p - 1).
    """
    result = 1
    for p, e in factors.items():
        result *= (p**(e + 1) - 1) // (p - 1)
    return result

def list_divisors(n):
    """Return sorted list of all divisors of n. O(sqrt(n))."""
    divs = []
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

def list_divisors_from_factors(factors):
    """Generate all divisors from factorization. O(d(n))."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return sorted(divs)

# Example:
# f = factorize(12)        -> {2:2, 3:1}
# num_divisors(f)          -> 6   (1,2,3,4,6,12)
# sum_divisors(f)          -> 28  (1+2+3+4+6+12)
# list_divisors(12)        -> [1, 2, 3, 4, 6, 12]


# ── Precomputed Divisor Count / Sum via Sieve ─────────────────────────────────
def divisor_sieve(n):
    """
    Compute num_divisors and sum_divisors for all i in [1, n] in O(n log n).
    Returns (d, sigma) arrays.
    """
    d = [0] * (n + 1)       # d[i] = number of divisors
    sigma = [0] * (n + 1)   # sigma[i] = sum of divisors
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            d[j] += 1
            sigma[j] += i
    return d, sigma

# Example:
# d, sigma = divisor_sieve(12)
# d[12] -> 6,  sigma[12] -> 28


# ── Euler's Totient Function ──────────────────────────────────────────────────
def euler_totient(n):
    """
    phi(n) = count of integers in [1, n] coprime to n.
    phi(n) = n * product((1 - 1/p)) for each prime p | n.
    """
    result = n
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result

def totient_from_factors(factors):
    """phi(n) from precomputed factorization. Faster if factors already known."""
    result = 1
    for p, e in factors.items():
        result *= (p - 1) * p**(e - 1)
    return result

# Example:
# euler_totient(12) -> 4   (1, 5, 7, 11 are coprime to 12)
# euler_totient(10) -> 4


# ── Perfect / Abundant / Deficient Numbers ────────────────────────────────────
def classify_number(n):
    """
    Perfect:    sigma(n) = 2n  (e.g., 6, 28, 496)
    Abundant:   sigma(n) > 2n  (e.g., 12)
    Deficient:  sigma(n) < 2n  (e.g., 8)
    """
    s = sum_divisors(factorize(n))
    if s == 2 * n: return "perfect"
    if s > 2 * n:  return "abundant"
    return "deficient"

# Example:
# classify_number(6)  -> "perfect"
# classify_number(12) -> "abundant"
# classify_number(8)  -> "deficient"
