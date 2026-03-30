"""
NAME: Combinatorics (nCr, Pascal's Triangle, Catalan, Stirling Numbers)
TAGS: combinatorics, binomial-coefficient, catalan, stirling, pascal
DESCRIPTION: Comprehensive combinatorics toolkit for competitions. Covers binomial coefficients
    with modular arithmetic, Pascal's triangle for small values, Catalan numbers for counting
    BSTs/paths/parenthesizations, and Stirling numbers for partition counting problems.
COMPLEXITY: Time O(N) precomputation for nCr, O(N^2) Pascal/Stirling; Space O(N) or O(N^2)
"""

MOD = 10**9 + 7


# ── Precomputed Factorials for Fast nCr ───────────────────────────────────────
def precompute(n, mod=MOD):
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], mod - 2, mod)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod
    return fact, inv_fact

MAXN = 300001
fact, inv_fact = precompute(MAXN)

def C(n, r, mod=MOD):
    """Binomial coefficient C(n, r) mod p in O(1). Requires precomputed tables."""
    if r < 0 or r > n or n < 0: return 0
    return fact[n] * inv_fact[r] % mod * inv_fact[n - r] % mod

def P(n, r, mod=MOD):
    """Permutation P(n, r) = n!/(n-r)! mod p in O(1)."""
    if r < 0 or r > n or n < 0: return 0
    return fact[n] * inv_fact[n - r] % mod

# Example:
# C(10, 3) -> 120
# C(100, 50) -> (large number mod MOD)


# ── Pascal's Triangle ─────────────────────────────────────────────────────────
def pascal(n, mod=None):
    """
    Build Pascal's triangle up to row n (0-indexed).
    pascal[i][j] = C(i, j).  Good for small n (n <= 2000).
    """
    tri = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        tri[i][0] = 1
        for j in range(1, i + 1):
            tri[i][j] = tri[i-1][j-1] + tri[i-1][j]
            if mod:
                tri[i][j] %= mod
    return tri

# Example:
# t = pascal(5)
# t[5][2] -> 10  (C(5,2))
# t[4][2] -> 6   (C(4,2))


# ── Catalan Numbers ───────────────────────────────────────────────────────────
def catalan(n, mod=MOD):
    """
    C_n = C(2n, n) / (n+1)
    Applications:
    - Number of valid parenthesizations of n+1 factors
    - Number of BSTs with n keys
    - Number of monotonic paths from (0,0) to (n,n) not crossing diagonal
    - Number of triangulations of a polygon with n+2 vertices
    Returns C_0, C_1, ..., C_n.
    """
    cats = [0] * (n + 1)
    for i in range(n + 1):
        cats[i] = C(2 * i, i, mod) * pow(i + 1, mod - 2, mod) % mod
    return cats

def catalan_dp(n, mod=MOD):
    """Catalan numbers via DP: C_n = sum(C_i * C_{n-1-i}) for i in [0, n-1]."""
    cats = [0] * (n + 1)
    cats[0] = 1
    for i in range(1, n + 1):
        for j in range(i):
            cats[i] = (cats[i] + cats[j] * cats[i-1-j]) % mod
    return cats

# Example:
# catalan(6) -> [1, 1, 2, 5, 14, 42, 132]
# C_5 = 42 (number of valid bracket sequences of length 10)


# ── Stars and Bars ────────────────────────────────────────────────────────────
def stars_and_bars(n, k, mod=MOD):
    """
    Number of ways to distribute n identical items into k distinct bins.
    = C(n + k - 1, k - 1)
    """
    return C(n + k - 1, k - 1, mod)

# Example:
# stars_and_bars(5, 3) -> C(7,2) = 21  (5 items in 3 bins)


# ── Stirling Numbers of the Second Kind ───────────────────────────────────────
def stirling_second(n, mod=MOD):
    """
    S(n, k) = number of ways to partition n elements into k non-empty subsets.
    S[i][j] = j * S[i-1][j] + S[i-1][j-1]
    Returns 2D table S[0..n][0..n].
    """
    S = [[0] * (n + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            S[i][j] = (j * S[i-1][j] + S[i-1][j-1]) % mod
    return S

# Example:
# S = stirling_second(5)
# S[5][2] -> 15  (ways to partition 5 elements into 2 non-empty subsets)
# S[4][2] -> 7


# ── Stirling Numbers of the First Kind ────────────────────────────────────────
def stirling_first(n, mod=MOD):
    """
    |s(n, k)| = unsigned Stirling numbers of the first kind.
    = number of permutations of n elements with exactly k cycles.
    |s[i][j]| = (i-1) * |s[i-1][j]| + |s[i-1][j-1]|
    """
    s = [[0] * (n + 1) for _ in range(n + 1)]
    s[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            s[i][j] = ((i - 1) * s[i-1][j] + s[i-1][j-1]) % mod
    return s

# Example:
# s = stirling_first(4)
# s[4][2] -> 11  (permutations of 4 elements with exactly 2 cycles)


# ── Multinomial Coefficient ───────────────────────────────────────────────────
def multinomial(n, groups, mod=MOD):
    """
    n! / (g1! * g2! * ... * gk!) where groups = [g1, g2, ..., gk], sum(groups) = n.
    Number of ways to divide n items into groups of sizes g1, g2, ..., gk.
    """
    assert sum(groups) == n
    result = fact[n]
    for g in groups:
        result = result * inv_fact[g] % mod
    return result

# Example:
# multinomial(6, [2, 2, 2]) -> 90  (ways to split 6 items into 3 pairs)
# multinomial(4, [1, 1, 2]) -> 12


# ── Derangements ──────────────────────────────────────────────────────────────
def derangements(n, mod=MOD):
    """
    D(n) = number of permutations with no fixed point.
    D(n) = (n-1) * (D(n-1) + D(n-2))
    Also: D(n) = n! * sum_{i=0}^{n} (-1)^i / i!
    """
    if n == 0: return 1
    if n == 1: return 0
    d = [0] * (n + 1)
    d[0], d[1] = 1, 0
    for i in range(2, n + 1):
        d[i] = (i - 1) * (d[i-1] + d[i-2]) % mod
    return d[n]

# Example:
# derangements(4) -> 9  (9 permutations of [1,2,3,4] with no fixed points)
# derangements(3) -> 2  ([2,3,1] and [3,1,2])


# ── Bell Numbers ─────────────────────────────────────────────────────────────
def bell_numbers(n, mod=MOD):
    """
    B(n) = total number of partitions of n elements into any number of subsets.
    B(n) = sum_{k=0}^{n} S(n, k)  (sum of Stirling numbers of second kind)
    """
    S = stirling_second(n, mod)
    return [sum(S[i]) % mod for i in range(n + 1)]

# Example:
# bell_numbers(4) -> [1, 1, 2, 5, 15]
# B(4) = 15: partitions of {1,2,3,4}


# ── Inclusion-Exclusion Template ──────────────────────────────────────────────
def inclusion_exclusion_example(n, primes):
    """
    Count integers in [1, n] NOT divisible by any prime in primes.
    Classic inclusion-exclusion: subtract, add back, subtract, ...
    """
    from itertools import combinations
    total = 0
    for size in range(len(primes) + 1):
        for combo in combinations(primes, size):
            product = 1
            for p in combo:
                product *= p
            sign = (-1) ** size
            total += sign * (n // product)
    return total

# Example:
# Integers in [1, 30] not divisible by 2, 3, or 5:
# inclusion_exclusion_example(30, [2, 3, 5]) -> 8
