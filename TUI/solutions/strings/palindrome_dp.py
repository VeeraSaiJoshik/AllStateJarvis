"""
NAME: Palindrome DP — Minimum Cuts and Counting
TAGS: string, DP, palindrome, palindrome partitioning, interval DP, counting
DESCRIPTION: Covers two classic palindrome DP problems: (1) minimum number of cuts to
             partition a string into palindromic substrings, solved in O(n^2) with a
             clever DP, and (2) counting all palindromic substrings in O(n^2) (or O(n)
             via Manacher). Also includes DP for counting palindromic subsequences.
             Essential for any competition problem involving palindrome decomposition.
COMPLEXITY: Min cuts O(n^2) time and space; Count substrings O(n^2) / O(n) with Manacher
"""

# --------------------------------------------------------------------------- #
#  Precompute is_palindrome table — O(n^2) time and space
# --------------------------------------------------------------------------- #

def build_palindrome_table(s: str) -> list[list[bool]]:
    """
    is_pal[i][j] = True iff s[i..j] (inclusive) is a palindrome.
    Used as a building block by many DP algorithms below.

    Example:
        s = "aab"
        is_pal[0][0]=T, is_pal[1][1]=T, is_pal[2][2]=T
        is_pal[0][1]=T  ("aa"), is_pal[1][2]=F, is_pal[0][2]=F
    """
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    # Every single character is a palindrome
    for i in range(n):
        is_pal[i][i] = True
    # Length-2 substrings
    for i in range(n - 1):
        is_pal[i][i + 1] = (s[i] == s[i + 1])
    # Lengths 3..n
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]
    return is_pal


# --------------------------------------------------------------------------- #
#  Minimum Palindrome Partition Cuts — O(n^2)
# --------------------------------------------------------------------------- #

def min_palindrome_cuts(s: str) -> int:
    """
    Returns the minimum number of cuts needed to partition s into palindromic
    substrings.  (Each piece must be a palindrome.)

    Example:
        min_palindrome_cuts("aab")      -> 1  ("aa" | "b")
        min_palindrome_cuts("a")        -> 0
        min_palindrome_cuts("ab")       -> 1  ("a" | "b")
        min_palindrome_cuts("aaaa")     -> 0  (whole string is palindrome)
        min_palindrome_cuts("abacaba")  -> 0
    """
    n = len(s)
    is_pal = build_palindrome_table(s)
    # dp[i] = min cuts for s[0..i]
    dp = [float('inf')] * n
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
        else:
            for j in range(1, i + 1):
                if is_pal[j][i]:
                    dp[i] = min(dp[i], dp[j - 1] + 1)
    return dp[n - 1]


def min_palindrome_cuts_partitions(s: str) -> tuple[int, list[str]]:
    """
    Returns (min_cuts, one optimal partition list).

    Example:
        min_palindrome_cuts_partitions("aab")  -> (1, ["aa", "b"])
    """
    n = len(s)
    is_pal = build_palindrome_table(s)
    dp = [float('inf')] * n
    parent = [-1] * n

    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
            parent[i] = -1
        else:
            for j in range(1, i + 1):
                if is_pal[j][i] and dp[j - 1] + 1 < dp[i]:
                    dp[i] = dp[j - 1] + 1
                    parent[i] = j

    # Reconstruct
    parts = []
    i = n - 1
    while i >= 0:
        j = parent[i]
        if j == -1:
            parts.append(s[0:i + 1])
            break
        else:
            parts.append(s[j:i + 1])
            i = j - 1
    parts.reverse()
    return dp[n - 1], parts


# --------------------------------------------------------------------------- #
#  Count all palindromic substrings — O(n^2)
# --------------------------------------------------------------------------- #

def count_palindromic_substrings_dp(s: str) -> int:
    """
    Counts all palindromic substrings (including single characters).
    Expands around each center (both odd and even lengths).

    Example:
        count_palindromic_substrings_dp("aaa")   -> 6
        count_palindromic_substrings_dp("abc")   -> 3
        count_palindromic_substrings_dp("abba")  -> 6
    """
    n = len(s)
    count = 0

    def expand(l: int, r: int) -> int:
        c = 0
        while l >= 0 and r < n and s[l] == s[r]:
            c += 1
            l -= 1
            r += 1
        return c

    for i in range(n):
        count += expand(i, i)       # odd-length palindromes
        count += expand(i, i + 1)   # even-length palindromes
    return count


# --------------------------------------------------------------------------- #
#  All palindromic partitions  (backtracking, exponential — for small inputs)
# --------------------------------------------------------------------------- #

def all_palindrome_partitions(s: str) -> list[list[str]]:
    """
    Returns all ways to partition s into palindromic substrings.

    Example:
        all_palindrome_partitions("aab")
        -> [["a", "a", "b"], ["aa", "b"]]
    """
    n = len(s)
    is_pal = build_palindrome_table(s)
    result: list[list[str]] = []
    path: list[str] = []

    def backtrack(start: int):
        if start == n:
            result.append(path[:])
            return
        for end in range(start, n):
            if is_pal[start][end]:
                path.append(s[start:end + 1])
                backtrack(end + 1)
                path.pop()

    backtrack(0)
    return result


# --------------------------------------------------------------------------- #
#  Count palindromic subsequences  (not substrings) — O(n^2) DP
# --------------------------------------------------------------------------- #

def count_palindromic_subsequences(s: str) -> int:
    """
    Counts the number of non-empty palindromic subsequences of s (mod 10^9+7).
    DP recurrence:
        dp[i][j] = # palindromic subseqs of s[i..j]
    Uses the formula:
        if s[i] != s[j]:  dp[i][j] = dp[i+1][j] + dp[i][j-1] - dp[i+1][j-1]
        elif ...: (adjusted for double-counting)

    Example:
        count_palindromic_subsequences("aab")   -> 4  (a,a,b,aa)
        count_palindromic_subsequences("abc")   -> 3  (a,b,c)
        count_palindromic_subsequences("aaa")   -> 7  (a,a,a,aa,aa,aa,aaa)
    """
    MOD = 10 ** 9 + 7
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] + 1) % MOD
            else:
                dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]) % MOD
    return dp[0][n - 1] % MOD


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    assert min_palindrome_cuts("aab") == 1
    assert min_palindrome_cuts("a") == 0
    assert min_palindrome_cuts("ab") == 1
    assert min_palindrome_cuts("aaaa") == 0
    assert min_palindrome_cuts("abacaba") == 0

    cuts, parts = min_palindrome_cuts_partitions("aab")
    assert cuts == 1
    assert parts == ["aa", "b"]

    assert count_palindromic_substrings_dp("aaa") == 6
    assert count_palindromic_substrings_dp("abc") == 3
    assert count_palindromic_substrings_dp("abba") == 6

    partitions = all_palindrome_partitions("aab")
    assert sorted(map(tuple, partitions)) == sorted([("a", "a", "b"), ("aa", "b")])

    assert count_palindromic_subsequences("aab") == 4
    assert count_palindromic_subsequences("abc") == 3
    assert count_palindromic_subsequences("aaa") == 7

    print("All Palindrome DP tests passed.")
