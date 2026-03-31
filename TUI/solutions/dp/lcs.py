"""
NAME: Longest Common Subsequence (LCS)
TAGS: dp, strings, subsequence, lcs
DESCRIPTION: Finds the longest subsequence common to two sequences (not necessarily
    contiguous). Core primitive for diff tools, DNA alignment, and plagiarism detection.
    Use whenever a competition asks for matching elements in order across two sequences.
COMPLEXITY: Time O(mn), Space O(min(m,n)) [space-optimized]
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LCS — O(mn) time, O(mn) space (with reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

def lcs_table(s, t):
    """
    Builds the full LCS DP table.
    dp[i][j] = LCS length of s[:i] and t[:j].

    Returns the table (needed for reconstruction).
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def lcs_length(s, t):
    """
    Returns only the LCS length in O(min(m,n)) space.

    Example:
        lcs_length("ABCBDAB", "BDCAB") → 4  (BCAB or BDAB)
    """
    if len(s) < len(t):
        s, t = t, s  # ensure s is the longer string
    m, n = len(s), len(t)
    # Use two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def lcs_string(s, t):
    """
    Returns the actual LCS string via backtracking.

    Example:
        lcs_string("AGGTAB", "GXTXAYB") → "GTAB"
    """
    dp = lcs_table(s, t)
    i, j = len(s), len(t)
    result = []
    while i > 0 and j > 0:
        if s[i - 1] == t[j - 1]:
            result.append(s[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(result))


# ─────────────────────────────────────────────────────────────────────────────
# ALL LCS STRINGS — enumerate every optimal solution
# ─────────────────────────────────────────────────────────────────────────────

def lcs_all(s, t):
    """
    Returns a set of ALL LCS strings (can be exponential — use cautiously).

    Example:
        lcs_all("AGTGATG", "GTTAG") → {'GTAG', 'GTTG', 'GTAG'} (varies)
    """
    dp = lcs_table(s, t)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def backtrack(i, j):
        if i == 0 or j == 0:
            return {""}
        if s[i - 1] == t[j - 1]:
            return {sub + s[i - 1] for sub in backtrack(i - 1, j - 1)}
        result = set()
        if dp[i - 1][j] >= dp[i][j - 1]:
            result |= backtrack(i - 1, j)
        if dp[i][j - 1] >= dp[i - 1][j]:
            result |= backtrack(i, j - 1)
        return result

    return backtrack(len(s), len(t))


# ─────────────────────────────────────────────────────────────────────────────
# SHORTEST COMMON SUPERSEQUENCE (SCS) — closely related to LCS
# SCS length = m + n - LCS(s, t)
# ─────────────────────────────────────────────────────────────────────────────

def scs_length(s, t):
    """
    Returns the length of the shortest common supersequence.

    Example:
        scs_length("AGGTAB", "GXTXAYB") → 9
        # LCS="GTAB" len=4; SCS = 6+7-4 = 9
    """
    return len(s) + len(t) - lcs_length(s, t)


def scs_string(s, t):
    """
    Returns the actual SCS string.

    Example:
        scs_string("ABCBDAB", "BDCAB") → one of several valid SCS strings
    """
    dp = lcs_table(s, t)
    i, j = len(s), len(t)
    result = []
    while i > 0 and j > 0:
        if s[i - 1] == t[j - 1]:
            result.append(s[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            result.append(s[i - 1])
            i -= 1
        else:
            result.append(t[j - 1])
            j -= 1
    # append remaining characters
    while i > 0:
        result.append(s[i - 1])
        i -= 1
    while j > 0:
        result.append(t[j - 1])
        j -= 1
    return "".join(reversed(result))


# ─────────────────────────────────────────────────────────────────────────────
# LONGEST COMMON SUBSTRING (contiguous, different from subsequence)
# ─────────────────────────────────────────────────────────────────────────────

def longest_common_substring(s, t):
    """
    Returns (length, start_in_s, start_in_t) of the longest common substring.
    Uses O(min(m,n)) space rolling array.

    Example:
        longest_common_substring("ABABC", "BABCAB") → (4, 1, 0)  # "BABC"
    """
    m, n = len(s), len(t)
    if m < n:
        s, t = t, s
        m, n = n, m
        swapped = True
    else:
        swapped = False

    best_len = best_i = best_j = 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_i = i - best_len   # start index in s
                    best_j = j - best_len   # start index in t
            else:
                curr[j] = 0
        prev, curr = curr, [0] * (n + 1)

    if swapped:
        return best_len, best_j, best_i
    return best_len, best_i, best_j


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    s, t = "ABCBDAB", "BDCAB"
    print("LCS length:", lcs_length(s, t))         # 4
    print("LCS string:", lcs_string(s, t))         # BCAB or BDAB
    print("SCS length:", scs_length(s, t))         # 9
    print("SCS string:", scs_string(s, t))

    s2, t2 = "AGGTAB", "GXTXAYB"
    print("LCS (AGGTAB, GXTXAYB):", lcs_string(s2, t2))  # GTAB

    length, si, ti = longest_common_substring("ABABC", "BABCAB")
    print(f"Longest common substring: len={length} s[{si}:] t[{ti}:]")  # 4
