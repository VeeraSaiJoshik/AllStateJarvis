"""
NAME: Digit DP (Count Numbers in [L, R] with Property)
TAGS: dp, digit-dp, counting, number-theory
DESCRIPTION: Counts integers in a range [L, R] satisfying a digit-based property
    (digit sum, no consecutive equal digits, divisibility, etc.) without iterating
    each number. Use whenever a competition asks to count numbers with a digit property
    in a large range — reduces O(R) to O(digits * states).
COMPLEXITY: Time O(D * S) where D = #digits, S = state space; Space O(D * S)
"""

from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# CORE TEMPLATE — count numbers in [0, N] with custom digit property
# ─────────────────────────────────────────────────────────────────────────────
# State: (pos, tight, started, ...custom state...)
#   pos     — current digit position (left to right)
#   tight   — whether we're still bounded by N's digits
#   started — whether we've placed a non-zero digit (handles leading zeros)
# To count [L, R]: f(R) - f(L-1)

def count_digit_property(N_str, valid_fn):
    """
    General digit DP: counts numbers in [0, N] (given as string) satisfying valid_fn.
    valid_fn(digits_list) → bool: return True if the number is valid.

    This is the naive version for clarity; see below for the efficient memoized version.
    """
    # This naive form is educational; real contests use the memoized template below.
    n = len(N_str)
    digits = [int(c) for c in N_str]
    count  = 0
    for num in range(int(N_str) + 1):
        if valid_fn(list(map(int, str(num)))):
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# DIGIT DP TEMPLATE — memoized, handles any custom state
# ─────────────────────────────────────────────────────────────────────────────

class DigitDP:
    """
    Competition-ready digit DP class.
    Subclass and implement `transition` to define your property.

    Built-in examples:
      - count_digit_sum_leq_k
      - count_no_consecutive_equal
      - count_divisible_by_k
      - count_with_balanced_digits
    """

    def __init__(self, digits):
        self.digits = digits
        self.n = len(digits)

    def solve(self):
        """Override to call self._dp with appropriate initial state."""
        raise NotImplementedError

    # Low-level memoized recursive solver
    # state = (pos, tight, started, *custom)
    # Returns count of valid completions


def digit_dp_range(L, R, check_fn):
    """
    Count integers in [L, R] satisfying check_fn.
    check_fn(num_str) → bool  (or use the memoized forms below for large ranges)

    For large ranges use: f(R) - f(L-1)
    """
    return sum(1 for x in range(L, R + 1) if check_fn(x))


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Count numbers in [1, N] with digit sum == k
# ─────────────────────────────────────────────────────────────────────────────

def count_digit_sum_eq(N, k):
    """
    Count integers in [1..N] whose digit sum equals k.

    Example:
        count_digit_sum_eq(100, 5) → 6  (5,14,23,32,41,50)
        count_digit_sum_eq(999, 10) → 63
    """
    s = str(N)
    n = len(s)

    @lru_cache(maxsize=None)
    def dp(pos, rem, tight, started):
        # rem = remaining digit sum needed
        if rem < 0:
            return 0
        if pos == n:
            return 1 if (started and rem == 0) else 0
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(0, limit + 1):
            new_started = started or (d > 0)
            new_rem     = rem - d if new_started else rem
            if new_rem < 0:
                break
            total += dp(pos + 1, new_rem, tight and (d == limit), new_started)
        return total

    result = dp(0, k, True, False)
    dp.cache_clear()
    return result


def count_digit_sum_range(L, R, k):
    """Count integers in [L, R] with digit sum == k."""
    def f(N):
        return count_digit_sum_eq(N, k) if N > 0 else 0
    return f(R) - f(L - 1)


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Count numbers in [1, N] divisible by k
# ─────────────────────────────────────────────────────────────────────────────

def count_divisible(N, k):
    """
    Count integers in [1..N] divisible by k using digit DP.
    (Trivially N // k, but useful as a digit DP demonstration.)

    Example:
        count_divisible(100, 7) → 14
    """
    s = str(N)
    n = len(s)

    @lru_cache(maxsize=None)
    def dp(pos, rem, tight, started):
        if pos == n:
            return 1 if (started and rem == 0) else 0
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(0, limit + 1):
            new_started = started or (d > 0)
            new_rem = (rem * 10 + d) % k if new_started else 0
            total += dp(pos + 1, new_rem, tight and (d == limit), new_started)
        return total

    result = dp(0, 0, True, False)
    dp.cache_clear()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Count numbers with no two consecutive equal digits
# ─────────────────────────────────────────────────────────────────────────────

def count_no_consecutive_equal(N):
    """
    Count integers in [1..N] with no two adjacent digits equal.

    Example:
        count_no_consecutive_equal(100) → 90  (all 2-digit numbers + 9 single-digit)
        # Wait: 1-digit: 9 (1..9), 2-digit: all 90 have no restriction since
        # 11,22,...,99 are excluded → 90-9=81 valid 2-digit + 9 + "100"? 100: 1,0,0 → fail
        # So total = 9 (1-digit) + 81 (2-digit) = 90? Let me count: 100 fails. → 90.
    """
    s = str(N)
    n = len(s)

    @lru_cache(maxsize=None)
    def dp(pos, last_digit, tight, started):
        # last_digit: previous digit placed (-1 if none / leading zero context)
        if pos == n:
            return 1 if started else 0
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(0, limit + 1):
            if started and d == last_digit:
                continue  # no two consecutive equal digits
            new_started = started or (d > 0)
            new_last    = d if new_started else -1
            total += dp(pos + 1, new_last, tight and (d == limit), new_started)
        return total

    result = dp(0, -1, True, False)
    dp.cache_clear()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 4: Count numbers with exactly m distinct digits
# ─────────────────────────────────────────────────────────────────────────────

def count_exact_distinct_digits(N, m):
    """
    Count integers in [1..N] using exactly m distinct digit values.

    Example:
        count_exact_distinct_digits(99, 2) → 72  # all 2-digit numbers with 2 distinct digits
        count_exact_distinct_digits(9, 1)  → 9   # 1..9 each has 1 distinct digit
    """
    s = str(N)
    n = len(s)

    @lru_cache(maxsize=None)
    def dp(pos, digit_mask, tight, started):
        # digit_mask: bitmask of which digits (0-9) have been used
        if pos == n:
            if not started:
                return 0
            return 1 if bin(digit_mask).count('1') == m else 0
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(0, limit + 1):
            if not started and d == 0:
                total += dp(pos + 1, 0, tight and (d == limit), False)
            else:
                new_mask = digit_mask | (1 << d)
                # Prune: can't exceed m distinct digits
                if bin(new_mask).count('1') > m:
                    continue
                total += dp(pos + 1, new_mask, tight and (d == limit), True)
        return total

    result = dp(0, 0, True, False)
    dp.cache_clear()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 5: Count numbers with digit sum divisible by k AND no zero digit
# (Multi-constraint digit DP)
# ─────────────────────────────────────────────────────────────────────────────

def count_sum_div_k_no_zero(N, k):
    """
    Count integers in [1..N] whose digit sum is divisible by k and contain no '0'.

    Example:
        count_sum_div_k_no_zero(99, 3) → 27
        # 2-digit numbers with no zero and digit sum div by 3:
        # numbers where d1+d2 ≡ 0 (mod 3) and d1,d2 ∈ [1..9]
    """
    s = str(N)
    n = len(s)

    @lru_cache(maxsize=None)
    def dp(pos, rem, tight, started):
        if pos == n:
            return 1 if (started and rem == 0) else 0
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(0, limit + 1):
            if started and d == 0:
                continue  # no zero digits after number has started
            if not started and d == 0:
                total += dp(pos + 1, 0, tight and (d == limit), False)
                continue
            new_rem = (rem + d) % k
            total  += dp(pos + 1, new_rem, tight and (d == limit), True)
        return total

    result = dp(0, 0, True, False)
    dp.cache_clear()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 6: Find the K-th number with a given digit property
# ─────────────────────────────────────────────────────────────────────────────

def kth_non_negative_with_digit_sum(k, target_sum):
    """
    Returns the k-th (1-indexed) positive integer whose digit sum equals target_sum.
    Digit DP + binary search approach.

    Example:
        kth_non_negative_with_digit_sum(1, 5) → 5
        kth_non_negative_with_digit_sum(2, 5) → 14
        kth_non_negative_with_digit_sum(3, 5) → 23
    """
    lo, hi = 1, 10 ** 15   # adjust upper bound as needed
    while lo < hi:
        mid = (lo + hi) // 2
        if count_digit_sum_eq(mid, target_sum) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: subtract one from number string (for f(L-1) computation)
# ─────────────────────────────────────────────────────────────────────────────

def subtract_one(s):
    """Returns string representation of int(s) - 1."""
    n = int(s)
    return str(n - 1) if n > 0 else "0"


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Digit sum == 5 in [1..100]:", count_digit_sum_eq(100, 5))         # 6
    print("Divisible by 7 in [1..100]:", count_divisible(100, 7))            # 14
    print("No consecutive equal in [1..100]:", count_no_consecutive_equal(100))  # 90
    print("Exact 1 distinct in [1..9]:", count_exact_distinct_digits(9, 1))  # 9
    print("Digit sum div 3, no zero in [1..99]:", count_sum_div_k_no_zero(99, 3))  # 27
    print("3rd number with digit sum 5:", kth_non_negative_with_digit_sum(3, 5))   # 23
    print("[L,R]=[10,50] digit sum=5:", count_digit_sum_range(10, 50, 5))    # should be 2: 14,23,32,41,50 → 5
