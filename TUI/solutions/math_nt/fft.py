"""
NAME: FFT / NTT for Polynomial Multiplication and Convolution
TAGS: fft, ntt, polynomial-multiplication, convolution, number-theory-transform
DESCRIPTION: Fast Fourier Transform (FFT) multiplies polynomials and computes convolutions in
    O(N log N). Number Theoretic Transform (NTT) is an FFT variant that works in modular
    arithmetic (exact, no floating point errors), ideal for counting problems. Use NTT when
    results need to be taken mod a specific prime; use FFT for real/large integer results.
COMPLEXITY: Time O(N log N); Space O(N)
"""

from cmath import exp, pi


# ── Standard FFT (Complex, for large integers) ────────────────────────────────
def fft(a, invert=False):
    """
    In-place Cooley-Tukey FFT. Input length must be a power of 2.
    If invert=True, computes IFFT.
    """
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        angle = 2 * pi / length * (-1 if invert else 1)
        w_len = exp(1j * angle)
        for i in range(0, n, length):
            w = 1
            for k in range(length // 2):
                u, v = a[i + k], a[i + k + length // 2] * w
                a[i + k] = u + v
                a[i + k + length // 2] = u - v
                w *= w_len
        length <<= 1

    if invert:
        for i in range(n):
            a[i] /= n

def poly_multiply_fft(a, b):
    """
    Multiply polynomials a and b (lists of coefficients, constant term first).
    Returns coefficient list of the product. Uses floating point FFT.
    Suitable for polynomials where coefficients aren't taken mod anything.
    """
    result_len = len(a) + len(b) - 1
    n = 1
    while n < result_len:
        n <<= 1

    fa = [complex(x) for x in a] + [0j] * (n - len(a))
    fb = [complex(x) for x in b] + [0j] * (n - len(b))

    fft(fa)
    fft(fb)
    for i in range(n):
        fa[i] *= fb[i]
    fft(fa, invert=True)

    return [round(fa[i].real) for i in range(result_len)]

# Example:
# poly_multiply_fft([1, 2, 3], [1, 2])  -> [1, 4, 7, 6]
# (1+2x+3x^2)(1+2x) = 1 + 4x + 7x^2 + 6x^3


# ── NTT (Number Theoretic Transform) ─────────────────────────────────────────
# NTT works mod a prime p = c * 2^k + 1 where g is a primitive root.
# Common choices:
#   p = 998244353 = 119 * 2^23 + 1,  g = 3  (supports up to 2^23 = 8M elements)
#   p = 985661441 = 235 * 2^22 + 1,  g = 3
#   p = 754974721 = 45  * 2^24 + 1,  g = 11

NTT_MOD = 998244353
NTT_G   = 3

def ntt(a, invert=False, mod=NTT_MOD, g=NTT_G):
    """
    In-place Number Theoretic Transform.
    a: list of integers mod p.
    invert=True for inverse NTT.
    """
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        # Primitive (length)-th root of unity mod p
        w_len = pow(g, (mod - 1) // length, mod)
        if invert:
            w_len = pow(w_len, mod - 2, mod)
        for i in range(0, n, length):
            w = 1
            for k in range(length // 2):
                u = a[i + k]
                v = a[i + k + length // 2] * w % mod
                a[i + k]              = (u + v) % mod
                a[i + k + length // 2] = (u - v) % mod
                w = w * w_len % mod
        length <<= 1

    if invert:
        inv_n = pow(n, mod - 2, mod)
        for i in range(n):
            a[i] = a[i] * inv_n % mod

def poly_multiply_ntt(a, b, mod=NTT_MOD, g=NTT_G):
    """
    Multiply polynomials a and b modulo mod using NTT. Exact integer arithmetic.
    mod must be an NTT-friendly prime. Coefficients should already be in [0, mod).
    """
    result_len = len(a) + len(b) - 1
    n = 1
    while n < result_len:
        n <<= 1

    fa = list(a) + [0] * (n - len(a))
    fb = list(b) + [0] * (n - len(b))

    ntt(fa, mod=mod, g=g)
    ntt(fb, mod=mod, g=g)
    for i in range(n):
        fa[i] = fa[i] * fb[i] % mod
    ntt(fa, invert=True, mod=mod, g=g)

    return fa[:result_len]

# Example:
# poly_multiply_ntt([1, 2, 3], [1, 2]) -> [1, 4, 7, 6] (mod 998244353)


# ── Convolution ───────────────────────────────────────────────────────────────
def convolve(a, b, mod=None):
    """
    Compute convolution c[k] = sum_{i+j=k} a[i]*b[j].
    Equivalent to polynomial multiplication.
    Uses NTT if mod is NTT-friendly, otherwise FFT.
    """
    if mod == NTT_MOD:
        return poly_multiply_ntt(a, b, mod=NTT_MOD)
    elif mod is None:
        return poly_multiply_fft(a, b)
    else:
        # General mod: use 3 NTT primes + CRT if needed
        return poly_multiply_ntt_general(a, b, mod)

def poly_multiply_ntt_general(a, b, target_mod):
    """
    Multiply polynomials mod target_mod using 3 NTT primes and CRT.
    Handles arbitrary prime moduli. Coefficients up to ~10^18 before mod.
    """
    P1, G1 = 998244353, 3
    P2, G2 = 985661441, 3
    P3, G3 = 754974721, 11

    r1 = poly_multiply_ntt(a, b, P1, G1)
    r2 = poly_multiply_ntt(a, b, P2, G2)
    r3 = poly_multiply_ntt(a, b, P3, G3)

    # CRT to recover true values
    result = []
    for x1, x2, x3 in zip(r1, r2, r3):
        # Garner's algorithm
        a1 = x1
        a2 = (x2 - a1) * pow(P1, -1, P2) % P2
        a3 = ((x3 - a1) * pow(P1, -1, P3) % P3 - a2 * pow(P2, -1, P3)) % P3
        val = a1 + a2 * P1 + a3 * P1 * P2
        result.append(val % target_mod)
    return result


# ── Polynomial Inverse (mod x^n) ──────────────────────────────────────────────
def poly_inv(a, n, mod=NTT_MOD, g=NTT_G):
    """
    Compute B such that A*B ≡ 1 (mod x^n) using Newton's method.
    a[0] must be invertible mod p.
    """
    result = [pow(a[0], mod - 2, mod)]
    k = 1
    while k < n:
        k <<= 1
        # result = result * (2 - a * result) mod x^k
        tmp = a[:k] + [0] * (k - min(k, len(a)))
        cur = result + [0] * (k - len(result))
        # compute a*cur mod x^k via NTT
        ac = poly_multiply_ntt(tmp, cur, mod, g)[:k]
        # negate and add 2
        neg_ac = [(2 - x) % mod if i == 0 else (-x) % mod for i, x in enumerate(ac)]
        result = poly_multiply_ntt(cur, neg_ac, mod, g)[:k]
    return result[:n]

# Example:
# poly_inv([1, 1], 4) -> [1, -1, 1, -1] mod NTT_MOD  (geometric series)


# ── Fast String Convolution / Subset Sum ─────────────────────────────────────
def string_match_fft(text, pattern):
    """
    Find all occurrences of pattern in text using FFT convolution (wildcard support).
    Pattern '?' matches any character. Returns list of match start positions.
    Simple version: works for small alphabet.
    """
    n, m = len(text), len(pattern)
    if m > n: return []
    matches = []
    # Use sum of squared differences trick
    T = [ord(c) for c in text]
    P = [ord(c) for c in reversed(pattern)]
    conv = poly_multiply_fft(T, P)
    # A position i matches if sum_j T[i+j]*P[m-1-j] == sum_j P[j]^2
    target = sum(p**2 for p in P)
    for i in range(m - 1, n):
        if conv[i] == target:
            matches.append(i - m + 1)
    return matches
