"""
NAME: Fast Power (Binary Exponentiation) and Matrix Exponentiation
TAGS: fast-power, modular-arithmetic, matrix-exponentiation, exponentiation-by-squaring
DESCRIPTION: Computes a^b mod m in O(log b) using repeated squaring. Matrix exponentiation
    extends this to compute matrix powers in O(k^3 log n) where k is matrix size, enabling
    O(log n) evaluation of linear recurrences. Use whenever you see large exponents or recurrences.
COMPLEXITY: Time O(log b) scalar, O(k^3 log n) matrix; Space O(k^2)
"""

# ── Scalar Fast Power ─────────────────────────────────────────────────────────
def power(base, exp, mod=None):
    """
    Computes base^exp (% mod if given).
    Python's built-in pow(base, exp, mod) is faster for modular; use this as reference.
    """
    result = 1
    base = base % mod if mod else base
    while exp > 0:
        if exp & 1:
            result = result * base % mod if mod else result * base
        base = base * base % mod if mod else base * base
        exp >>= 1
    return result

# Prefer the built-in:  pow(base, exp, mod)  — it's implemented in C.

# Example:
# power(2, 10, 1000) -> 24
# pow(2, 10, 1000)   -> 24  (same, faster)
# pow(2, -1, MOD)    -> modular inverse (Python 3.8+)


# ── Matrix Multiplication ─────────────────────────────────────────────────────
def mat_mul(A, B, mod=None):
    """Multiply two square matrices A and B (mod m if given)."""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if A[i][k] == 0:
                continue
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
        if mod:
            for j in range(n):
                C[i][j] %= mod
    return C

def mat_pow(M, p, mod=None):
    """Compute matrix M^p (mod m if given) using fast exponentiation."""
    n = len(M)
    # Identity matrix
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    while p > 0:
        if p & 1:
            result = mat_mul(result, M, mod)
        M = mat_mul(M, M, mod)
        p >>= 1
    return result


# ── Fibonacci via Matrix Exponentiation ───────────────────────────────────────
def fibonacci(n, mod=None):
    """
    Compute F(n) in O(log n) using matrix exponentiation.
    F(0)=0, F(1)=1, F(2)=1, F(3)=2, ...
    [[1,1],[1,0]]^n = [[F(n+1), F(n)], [F(n), F(n-1)]]
    """
    if n == 0: return 0
    M = [[1, 1], [1, 0]]
    result = mat_pow(M, n, mod)
    return result[0][1]

# Example:
# fibonacci(10)          -> 55
# fibonacci(10**18, MOD) -> answer mod MOD


# ── Generic Linear Recurrence ─────────────────────────────────────────────────
def linear_recurrence(coeffs, init, n, mod):
    """
    Compute a[n] for a linear recurrence defined by:
        a[i] = coeffs[0]*a[i-1] + coeffs[1]*a[i-2] + ... + coeffs[k-1]*a[i-k]
    init = [a[0], a[1], ..., a[k-1]]  (first k terms)
    coeffs = [c1, c2, ..., ck]

    Example: Tribonacci: coeffs=[1,1,1], init=[0,1,1]
    """
    k = len(coeffs)
    if n < k:
        return init[n] % mod

    # Companion matrix
    # [a[i]  ]   [c1 c2 ... ck] [a[i-1]  ]
    # [a[i-1]] = [1  0  ... 0 ] [a[i-2]  ]
    # [  ...  ]   [0  1  ... 0 ] [  ...   ]
    # [a[i-k+1]  [0  0  ... 0 ] [a[i-k]  ]
    M = [[0] * k for _ in range(k)]
    for j in range(k):
        M[0][j] = coeffs[j] % mod
    for i in range(1, k):
        M[i][i - 1] = 1

    Mp = mat_pow(M, n - k + 1, mod)

    # State vector: [a[k-1], a[k-2], ..., a[0]]
    ans = 0
    for j in range(k):
        ans = (ans + Mp[0][j] * init[k - 1 - j]) % mod
    return ans

# Example:
# Fibonacci: linear_recurrence([1,1], [0,1], 10, 10**9+7) -> 55
# Tribonacci: linear_recurrence([1,1,1], [0,1,1], 10, 10**9+7) -> 149


# ── Useful Patterns ───────────────────────────────────────────────────────────
# Geometric series sum: 1 + a + a^2 + ... + a^(n-1) mod p
# Use matrix: [[a, 0], [1, 1]]^n gives [[a^n, 0], [(a^n-1)/(a-1), 1]]
# Or use: (pow(a, n, p) - 1) * pow(a - 1, p-2, p) % p  (when a != 1)
def geometric_series_sum(a, n, mod):
    """Sum of a^0 + a^1 + ... + a^(n-1) mod p."""
    if a % mod == 1:
        return n % mod
    return (pow(a, n, mod) - 1) * pow(a - 1, mod - 2, mod) % mod

# Example: geometric_series_sum(2, 4, 10**9+7) -> 15  (1+2+4+8)
