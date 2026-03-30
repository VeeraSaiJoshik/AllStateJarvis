"""
NAME: Modular Arithmetic (Inverse, Division, Factorials, Square Root)
TAGS: modular-arithmetic, modular-inverse, factorial, number-theory
DESCRIPTION: Core modular arithmetic toolkit for competitive programming. Covers modular
    inverse via Fermat's little theorem (for prime moduli) and extended GCD (for any modulus),
    plus O(N) precomputed factorial and inverse-factorial tables for fast nCr queries. Also
    includes Tonelli-Shanks modular square root for quadratic residue problems.
COMPLEXITY: Time O(log MOD) per inverse, O(N) precomputation; Space O(N)
"""

MOD = 10**9 + 7   # Standard prime modulus (change as needed)


# ── Modular Inverse ───────────────────────────────────────────────────────────
def inv_fermat(a, mod=MOD):
    """
    Modular inverse using Fermat's little theorem: a^(p-1) ≡ 1 (mod p).
    So a^(-1) ≡ a^(p-2) (mod p).  REQUIRES mod to be prime.
    """
    return pow(a, mod - 2, mod)

def inv_ext_gcd(a, mod):
    """
    Modular inverse using extended Euclidean. Works for any coprime (a, mod).
    Equivalent to pow(a, -1, mod) in Python 3.8+.
    """
    g, x, _ = _ext_gcd(a % mod, mod)
    if g != 1:
        raise ValueError(f"gcd({a}, {mod}) = {g} != 1, inverse doesn't exist")
    return x % mod

def _ext_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = _ext_gcd(b, a % b)
    return g, y, x - (a // b) * y

# Python 3.8+ shortcut (preferred in contests):
# pow(a, -1, mod)  ->  modular inverse when gcd(a, mod) = 1

# Example:
# inv_fermat(3, 10**9+7)  -> 333333336  (3 * 333333336 ≡ 1 mod 10^9+7)
# pow(3, -1, 10**9+7)     -> 333333336


# ── Precomputed Factorial & Inverse Factorial Table ───────────────────────────
def precompute_factorials(n, mod=MOD):
    """
    Returns (fact, inv_fact) arrays of size n+1.
    fact[i]     = i! mod p
    inv_fact[i] = (i!)^(-1) mod p
    Enables O(1) nCr, nPr queries.
    """
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % mod

    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], mod - 2, mod)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod

    return fact, inv_fact

# Usage pattern for competitions:
# MAXN = 2 * 10**6
# fact, inv_fact = precompute_factorials(MAXN)
# def C(n, r): return fact[n] * inv_fact[r] % MOD * inv_fact[n-r] % MOD if 0<=r<=n else 0


# ── nCr with precomputed tables ───────────────────────────────────────────────
MAXN = 300001
fact, inv_fact = precompute_factorials(MAXN)

def comb(n, r, mod=MOD):
    """C(n, r) mod p using precomputed factorials. O(1) per query."""
    if r < 0 or r > n:
        return 0
    return fact[n] * inv_fact[r] % mod * inv_fact[n - r] % mod

def perm(n, r, mod=MOD):
    """P(n, r) = n!/(n-r)! mod p. O(1) per query."""
    if r < 0 or r > n:
        return 0
    return fact[n] * inv_fact[n - r] % mod

# Example:
# comb(10, 3)   -> 120
# perm(10, 3)   -> 720


# ── Precomputed Inverse Array ─────────────────────────────────────────────────
def precompute_inverses(n, mod=MOD):
    """
    Compute inv[i] for i in [1..n] in O(n).
    Uses the recurrence: inv[i] = -(mod // i) * inv[mod % i] % mod
    Useful for harmonic series computations and online inverse queries.
    """
    inv = [0] * (n + 1)
    inv[1] = 1
    for i in range(2, n + 1):
        inv[i] = -(mod // i) * inv[mod % i] % mod
    return inv

# Example:
# inv = precompute_inverses(10)
# inv[3] -> 333333336  (mod 10^9+7)
# inv[7] -> 142857144


# ── Modular Division ──────────────────────────────────────────────────────────
def mod_div(a, b, mod=MOD):
    """a / b mod p.  Requires gcd(b, p) = 1."""
    return a * pow(b, mod - 2, mod) % mod

# Example: mod_div(10, 4, 10**9+7) -> 10 * inv(4) mod p


# ── Lucas' Theorem (nCr for large n, small prime mod) ─────────────────────────
def lucas_comb(n, r, p):
    """
    C(n, r) mod p using Lucas' theorem. Works for prime p even when n > p.
    Decomposes n and r in base p, then multiplies C(n_i, r_i) for each digit pair.
    Time: O(log_p(n) * p) for table build + O(log_p(n)) per query.
    """
    if r == 0:
        return 1
    if n < r:
        return 0

    # Precompute small factorials up to p
    f = [1] * (p + 1)
    for i in range(1, p + 1):
        f[i] = f[i - 1] * i % p
    inv_f = [1] * (p + 1)
    inv_f[p - 1] = pow(f[p - 1], p - 2, p)
    for i in range(p - 2, -1, -1):
        inv_f[i] = inv_f[i + 1] * (i + 1) % p

    def small_comb(a, b):
        if b < 0 or b > a: return 0
        return f[a] * inv_f[b] % p * inv_f[a - b] % p

    result = 1
    while n or r:
        ni, ri = n % p, r % p
        result = result * small_comb(ni, ri) % p
        n //= p
        r //= p
    return result

# Example:
# lucas_comb(10, 3, 7)   -> C(10,3) mod 7 = 120 mod 7 = 1
# lucas_comb(1000, 500, 13) -> C(1000,500) mod 13


# ── Modular Square Root (Tonelli-Shanks) ──────────────────────────────────────
def sqrt_mod(n, p):
    """
    Find x such that x^2 ≡ n (mod p), p is prime.
    Returns x (the smaller root) or None if n is not a quadratic residue mod p.
    The other root is p - x.
    """
    n %= p
    if n == 0: return 0
    if pow(n, (p - 1) // 2, p) != 1:
        return None  # Not a quadratic residue

    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    # Tonelli-Shanks algorithm
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find a non-residue z
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m  = s
    c  = pow(z, q, p)
    t  = pow(n, q, p)
    r  = pow(n, (q + 1) // 2, p)

    while True:
        if t == 1:
            return min(r, p - r)
        i, tmp = 1, t * t % p
        while tmp != 1:
            tmp = tmp * tmp % p
            i += 1
        b  = pow(c, 1 << (m - i - 1), p)
        m  = i
        c  = b * b % p
        t  = t * c % p
        r  = r * b % p

# Example:
# sqrt_mod(2, 7)  -> 3  (3^2 = 9 ≡ 2 mod 7; also p-3=4, 4^2=16≡2 mod 7)
# sqrt_mod(3, 7)  -> None  (3 is not a quadratic residue mod 7)
# sqrt_mod(4, 13) -> 2


# ── Discrete Logarithm (Baby-Step Giant-Step) ─────────────────────────────────
def discrete_log(a, b, mod):
    """
    Find minimum x >= 0 such that a^x ≡ b (mod p), p prime.
    Uses Baby-Step Giant-Step in O(sqrt(p)) time and space.
    Returns x or None if no solution.
    """
    from math import isqrt
    n = isqrt(mod) + 1
    # Baby step: store a^j mod p for j in [0, n)
    table = {}
    aj = 1
    for j in range(n):
        table[b * aj % mod] = j
        aj = aj * a % mod

    # Giant step: check a^(i*n) for i in [1, n]
    an = pow(a, n, mod)
    cur = an
    for i in range(1, n + 2):
        if cur in table:
            ans = i * n - table[cur]
            if ans >= 0:
                return ans
        cur = cur * an % mod
    return None

# Example:
# discrete_log(2, 22, 29)  -> x such that 2^x ≡ 22 (mod 29)
# pow(2, discrete_log(2,22,29), 29) == 22
