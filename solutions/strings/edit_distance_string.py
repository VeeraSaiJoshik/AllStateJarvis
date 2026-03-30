"""
NAME: Edit Distance and Longest Common Substring (DP)
TAGS: string, DP, edit distance, Levenshtein, LCS substring, alignment
DESCRIPTION: Covers the classic string edit-distance (Levenshtein) with full DP table,
             path reconstruction for the edit sequence, and the longest common substring
             (contiguous, not subsequence) via DP. Also includes LCS subsequence length
             for completeness. Use in competitions involving string transformation costs,
             approximate matching, or shared content detection.
COMPLEXITY: Edit distance O(nm) time and space (O(min(n,m)) space with rolling array);
            Longest common substring O(nm) time, O(min(n,m)) space
"""

# --------------------------------------------------------------------------- #
#  Edit Distance (Levenshtein)
# --------------------------------------------------------------------------- #

def edit_distance(s: str, t: str) -> int:
    """
    Minimum number of single-character insert / delete / substitute operations
    to transform s into t.

    Example:
        edit_distance("kitten", "sitting")  -> 3
        edit_distance("", "abc")            -> 3
        edit_distance("abc", "abc")         -> 0
    """
    n, m = len(s), len(t)
    # Space-optimised: two rows
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j],      # delete
                                  curr[j - 1],   # insert
                                  prev[j - 1])   # substitute
        prev = curr
    return prev[m]


def edit_distance_full(s: str, t: str) -> tuple[int, list[str]]:
    """
    Returns (distance, operations) where operations is a list of human-readable
    edit steps to transform s into t.

    Operations: "MATCH i j", "REPLACE i j", "INSERT j", "DELETE i"

    Example:
        d, ops = edit_distance_full("cat", "cut")
        # d = 1, ops includes "REPLACE 1 1"  (change 'a' -> 'u')
    """
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],
                                   dp[i][j - 1],
                                   dp[i - 1][j - 1])

    # Backtrack
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i - 1] == t[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(f"MATCH {i-1} {j-1} ('{s[i-1]}')")
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"REPLACE {i-1} {j-1} ('{s[i-1]}'->'{t[j-1]}')")
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f"DELETE {i-1} ('{s[i-1]}')")
            i -= 1
        else:
            ops.append(f"INSERT {j-1} ('{t[j-1]}')")
            j -= 1
    ops.reverse()
    return dp[n][m], ops


# --------------------------------------------------------------------------- #
#  Longest Common Substring  (contiguous)
# --------------------------------------------------------------------------- #

def longest_common_substring_dp(s: str, t: str) -> str:
    """
    Returns the longest substring (contiguous) common to both s and t.
    Uses O(min(n,m)) space rolling DP.

    Example:
        longest_common_substring_dp("abcde", "bcdef")  -> "bcde"
        longest_common_substring_dp("abcd", "efgh")    -> ""
        longest_common_substring_dp("xyzabcxyz", "abc") -> "abc"
    """
    if len(s) < len(t):
        s, t = t, s         # make s the longer string
    n, m = len(s), len(t)
    best_len = 0
    best_end_s = 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_end_s = i
            else:
                curr[j] = 0
        prev = curr
    return s[best_end_s - best_len:best_end_s]


# --------------------------------------------------------------------------- #
#  Longest Common Subsequence  (non-contiguous, for reference)
# --------------------------------------------------------------------------- #

def lcs_length(s: str, t: str) -> int:
    """
    Length of the longest common subsequence (not necessarily contiguous).

    Example:
        lcs_length("abcde", "ace")   -> 3
        lcs_length("abc", "def")     -> 0
    """
    n, m = len(s), len(t)
    if n < m:
        s, t = t, s
        n, m = m, n
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


# --------------------------------------------------------------------------- #
#  Bonus: shortest edit script length  (= edit distance, here for clarity)
# --------------------------------------------------------------------------- #

def min_insertions_deletions(s: str, t: str) -> tuple[int, int]:
    """
    Returns (deletions, insertions) needed to transform s into t
    using only insert/delete (no substitution — a substitution costs 2).
    deletions = len(s) - lcs, insertions = len(t) - lcs.

    Example:
        min_insertions_deletions("heap", "pea")  -> (2, 1)
    """
    lcs = lcs_length(s, t)
    return len(s) - lcs, len(t) - lcs


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "") == 3

    d, ops = edit_distance_full("cat", "cut")
    assert d == 1

    assert longest_common_substring_dp("abcde", "bcdef") == "bcde"
    assert longest_common_substring_dp("abcd", "efgh") == ""
    assert longest_common_substring_dp("xyzabcxyz", "abc") == "abc"

    assert lcs_length("abcde", "ace") == 3
    assert lcs_length("abc", "def") == 0

    assert min_insertions_deletions("heap", "pea") == (2, 1)

    print("All Edit Distance / LCS tests passed.")
