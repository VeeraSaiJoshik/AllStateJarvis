"""
NAME: Edit Distance (Levenshtein Distance)
TAGS: dp, strings, edit-distance, levenshtein
DESCRIPTION: Computes the minimum number of single-character edits (insert, delete,
    replace) to transform one string into another. Use in competitions involving
    string similarity, DNA mutation counting, or spell-checking problems.
COMPLEXITY: Time O(mn), Space O(min(m,n)) [space-optimized]
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD EDIT DISTANCE — O(mn) time, O(min(m,n)) space
# ─────────────────────────────────────────────────────────────────────────────

def edit_distance(s, t, ins_cost=1, del_cost=1, rep_cost=1):
    """
    Returns the minimum weighted edit distance from s → t.
    Default: unit costs (Levenshtein distance).
    Customize ins/del/rep costs for weighted variants.

    Recurrence:
        dp[i][j] = edit distance between s[:i] and t[:j]
        if s[i-1]==t[j-1]:  dp[i][j] = dp[i-1][j-1]
        else:                dp[i][j] = min(
                                 dp[i-1][j]   + del_cost,   # delete from s
                                 dp[i][j-1]   + ins_cost,   # insert into s
                                 dp[i-1][j-1] + rep_cost    # replace
                             )

    Example:
        edit_distance("kitten", "sitting") → 3
        edit_distance("horse", "ros")      → 3
        edit_distance("", "abc")           → 3
    """
    # Ensure s is the longer string to minimize memory
    if len(s) < len(t):
        s, t = t, s
        ins_cost, del_cost = del_cost, ins_cost
    m, n = len(s), len(t)

    # dp[j] = edit distance between s[:current_row] and t[:j]
    prev = list(range(n + 1))  # base case: t[:j] needs j inserts from ""
    # scale by insert cost
    prev = [j * ins_cost for j in range(n + 1)]

    for i in range(1, m + 1):
        curr = [i * del_cost] + [0] * n   # delete i chars from s
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = min(
                    prev[j]     + del_cost,    # delete s[i-1]
                    curr[j - 1] + ins_cost,    # insert t[j-1]
                    prev[j - 1] + rep_cost,    # replace s[i-1] with t[j-1]
                )
        prev = curr
    return prev[n]


# ─────────────────────────────────────────────────────────────────────────────
# EDIT DISTANCE WITH BACKTRACKING — returns the list of operations
# ─────────────────────────────────────────────────────────────────────────────

def edit_distance_ops(s, t):
    """
    Returns (distance, operations_list).
    operations_list is a sequence of (op, char, position) tuples:
        ('keep',    c, i)
        ('replace', c_old→c_new, i)
        ('insert',  c, i)
        ('delete',  c, i)

    Example:
        edit_distance_ops("kitten", "sitting")
        → (3, [('replace','k→s',0),('replace','e→i',4),('insert','g',6)])
    """
    m, n = len(s), len(t)
    # Full 2D table for backtracking
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to find operations
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i - 1] == t[j - 1]:
            ops.append(('keep', s[i - 1], i - 1))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('replace', f'{s[i-1]}→{t[j-1]}', i - 1))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('delete', s[i - 1], i - 1))
            i -= 1
        else:
            ops.append(('insert', t[j - 1], j - 1))
            j -= 1
    ops.reverse()
    return dp[m][n], ops


# ─────────────────────────────────────────────────────────────────────────────
# EDIT DISTANCE VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

def hamming_distance(s, t):
    """
    Number of positions where s and t differ (both must have equal length).
    Special case of edit distance with only replacements allowed.

    Example:
        hamming_distance("karolin", "kathrin") → 3
    """
    assert len(s) == len(t), "Hamming distance requires equal-length strings"
    return sum(a != b for a, b in zip(s, t))


def lcs_edit_distance(s, t):
    """
    Edit distance using only inserts and deletes (no replacements).
    = len(s) + len(t) - 2 * LCS(s, t)

    Example:
        lcs_edit_distance("ABCDE", "ACE") → 2  # delete B and D
    """
    # Inline LCS to avoid import
    m, n = len(s), len(t)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    return (m - lcs) + (n - lcs)


def edit_distance_2d_full(s, t):
    """
    Returns the full O(mn) DP table.  Useful when you need dp[i][j] for all i,j
    (e.g., substring matching: find min edit distance from s to any substring of t).

    Substring trick: initialize dp[0][j] = 0 (can start matching anywhere in t).

    Example:
        # Find substring of t closest to s
        s, t = "abc", "xabcy"
        dp = edit_distance_2d_full(s, t)
        min_cost = min(dp[len(s)])  # → 0 (exact match exists)
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    # dp[0][j] = 0 for substring matching (start free), or j for full match
    for j in range(n + 1):
        dp[0][j] = j  # change to 0 for "s as substring of t" variant
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp


def min_edit_to_palindrome(s):
    """
    Minimum insertions to make string a palindrome.
    = LCS(s, reverse(s))... actually = len(s) - LCS(s, rev(s)).

    Example:
        min_edit_to_palindrome("abcd") → 3  # "abcdcba" needs 3 inserts
        min_edit_to_palindrome("abcba") → 0
    """
    t = s[::-1]
    m, n = len(s), len(t)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return len(s) - prev[n]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("kitten→sitting:", edit_distance("kitten", "sitting"))   # 3
    print("horse→ros:",      edit_distance("horse", "ros"))        # 3
    print("")
    dist, ops = edit_distance_ops("kitten", "sitting")
    print("ops:", dist, ops)
    print("")
    print("Hamming:", hamming_distance("karolin", "kathrin"))      # 3
    print("LCS edit (ABCDE, ACE):", lcs_edit_distance("ABCDE", "ACE"))  # 2
    print("Min inserts for palindrome 'abcd':", min_edit_to_palindrome("abcd"))  # 3
