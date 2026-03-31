"""
NAME: Matrix Exponentiation for Linear Recurrences
TAGS: matrix-exponentiation, linear-recurrence, fibonacci, dynamic-programming
DESCRIPTION: Evaluates any k-th order linear recurrence at position N in O(k^3 log N) time
    by expressing the recurrence as a matrix transition and using fast matrix power. Essential
    for Fibonacci-like problems with large N (up to 10^18) and for counting paths of length N
    in a fixed graph.
COMPLEXITY: Time O(k^3 log N) where k is the recurrence order; Space O(k^2)
"""

MOD = 10**9 + 7


# ── Core Matrix Operations ────────────────────────────────────────────────────
def mat_mul(A, B, mod):
    """Multiply two square matrices mod p. O(k^3)."""
    k = len(A)
    C = [[0] * k for _ in range(k)]
    for i in range(k):
        for l in range(k):
            if A[i][l] == 0:
                continue
            for j in range(k):
                C[i][j] += A[i][l] * B[l][j]
        for j in range(k):
            C[i][j] %= mod
    return C

def mat_pow(M, p, mod):
    """Compute M^p mod p using fast exponentiation. O(k^3 log p)."""
    k = len(M)
    # Identity matrix
    R = [[int(i == j) for j in range(k)] for i in range(k)]
    while p:
        if p & 1:
            R = mat_mul(R, M, mod)
        M = mat_mul(M, M, mod)
        p >>= 1
    return R

def mat_vec_mul(M, v, mod):
    """Multiply matrix M by column vector v. Returns result vector."""
    k = len(M)
    result = [0] * k
    for i in range(k):
        for j in range(k):
            result[i] = (result[i] + M[i][j] * v[j]) % mod
    return result


# ── Fibonacci Numbers ─────────────────────────────────────────────────────────
def fib(n, mod=MOD):
    """
    F(n) in O(log n).  F(0)=0, F(1)=1, F(2)=1, F(3)=2, ...
    Transition: [[1,1],[1,0]]^n  =>  [[F(n+1), F(n)], [F(n), F(n-1)]]
    """
    if n == 0: return 0
    if n == 1: return 1 % mod
    M = [[1, 1], [1, 0]]
    R = mat_pow(M, n - 1, mod)
    return R[0][0]

def fib_pair(n, mod=MOD):
    """Returns (F(n), F(n+1)) simultaneously, useful for further computation."""
    if n == 0: return (0, 1 % mod)
    M = [[1, 1], [1, 0]]
    R = mat_pow(M, n, mod)
    return R[1][0], R[0][0]   # F(n), F(n+1)

# Example:
# fib(10)          -> 55
# fib(10**18, MOD) -> some large answer mod MOD
# fib_pair(10)     -> (55, 89)


# ── General k-th Order Linear Recurrence ─────────────────────────────────────
def solve_recurrence(coeffs, init, n, mod=MOD):
    """
    Compute a[n] for the recurrence:
        a[i] = c[0]*a[i-1] + c[1]*a[i-2] + ... + c[k-1]*a[i-k]
    coeffs = [c[0], c[1], ..., c[k-1]]   (k coefficients)
    init   = [a[0], a[1], ..., a[k-1]]   (first k base cases)

    Companion (transition) matrix T:
        [ c[0]  c[1]  ...  c[k-1] ]
        [  1     0    ...    0    ]
        [  0     1    ...    0    ]
        [ ...                     ]
        [  0     0    ...    0    ]

    State vector: v = [a[i], a[i-1], ..., a[i-k+1]]^T
    Then T * v = next state.
    After n-k+1 applications: T^(n-k+1) * [a[k-1], ..., a[0]]^T
    """
    k = len(coeffs)
    if n < k:
        return init[n] % mod

    # Build companion matrix
    T = [[0] * k for _ in range(k)]
    for j in range(k):
        T[0][j] = coeffs[j] % mod
    for i in range(1, k):
        T[i][i - 1] = 1

    Tp = mat_pow(T, n - k + 1, mod)

    # Initial state vector: [a[k-1], a[k-2], ..., a[0]]
    v = [init[k - 1 - i] % mod for i in range(k)]
    result_vec = mat_vec_mul(Tp, v, mod)
    return result_vec[0]

# Example:
# Fibonacci:   solve_recurrence([1,1], [0,1], 10) -> 55
# Tribonacci:  solve_recurrence([1,1,1], [0,1,1], 10) -> 149
# Padovan:     solve_recurrence([0,1,1], [1,1,1], 10) -> a(10)


# ── Graph Path Counting ───────────────────────────────────────────────────────
def count_paths(adj_matrix, steps, mod=MOD):
    """
    Count paths of exactly `steps` steps in a graph.
    adj_matrix[i][j] = number of edges from i to j.
    Result matrix R = adj_matrix^steps; R[i][j] = paths from i to j.
    """
    return mat_pow(adj_matrix, steps, mod)

# Example: Count paths of length 3 in a 3-node graph
# adj = [[0,1,1],[1,0,1],[1,1,0]]
# R = count_paths(adj, 3)
# R[0][0] = number of cycles of length 3 starting from node 0


# ── Recurrence with Additive Constant ────────────────────────────────────────
def solve_affine_recurrence(a, b, x0, n, mod=MOD):
    """
    Solve: x[n] = a*x[n-1] + b  with x[0] = x0.
    Closed form: x[n] = a^n * x0 + b*(a^n - 1)/(a-1)  if a != 1
               : x[n] = x0 + n*b                        if a == 1
    Via matrix:  [[a, b], [0, 1]]^n * [x0, 1]^T
    """
    if a == 1:
        return (x0 + n * b) % mod
    M = [[a % mod, b % mod], [0, 1]]
    R = mat_pow(M, n, mod)
    return (R[0][0] * x0 + R[0][1]) % mod

# Example:
# solve_affine_recurrence(2, 1, 0, 5) -> 31  (0,1,3,7,15,31)
# solve_affine_recurrence(1, 3, 0, 4) -> 12  (0,3,6,9,12)


# ── Prebuilt Transition Matrices ──────────────────────────────────────────────
# Fibonacci sequence matrix
FIB_MATRIX = [[1, 1], [1, 0]]

# Tribonacci: a[n] = a[n-1] + a[n-2] + a[n-3]
TRIB_MATRIX = [[1, 1, 1], [1, 0, 0], [0, 1, 0]]

# Pell numbers: P[n] = 2*P[n-1] + P[n-2]
PELL_MATRIX = [[2, 1], [1, 0]]

# Jacobsthal: J[n] = J[n-1] + 2*J[n-2]
JACOBSTHAL_MATRIX = [[1, 2], [1, 0]]

# Usage:
# nth_fib = mat_pow(FIB_MATRIX, n, MOD)[0][1]
# nth_pell = mat_pow(PELL_MATRIX, n, MOD)[0][1]
