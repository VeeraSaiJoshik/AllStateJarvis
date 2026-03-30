"""
NAME: Manacher's Algorithm for Palindromic Substrings
TAGS: string, palindrome, Manacher, longest palindromic substring, O(n)
DESCRIPTION: Manacher's algorithm finds the longest palindromic substring and the radius
             of every palindrome centered at each position in O(n) time by reusing
             previously computed palindrome extents. Use it whenever a problem involves
             counting, finding, or querying palindromic substrings efficiently.
COMPLEXITY: Time O(n), Space O(n)
"""

# --------------------------------------------------------------------------- #
#  Core: build palindrome radius array on the transformed string
#  Transform: insert '#' between every character and at both ends.
#  "abc"  ->  "#a#b#c#"   (length 2n+1)
#  This unifies odd and even length palindromes.
# --------------------------------------------------------------------------- #

def _transform(s: str) -> str:
    return '#' + '#'.join(s) + '#'


def manacher(s: str) -> list[int]:
    """
    Returns the p-array on the TRANSFORMED string T = _transform(s).
    p[i] = radius of palindrome centered at T[i] (including center).

    To get the radius for original string:
      - Odd palindrome centered at s[i]:  p[2*i + 1] - 1  is the half-length
        (i.e. palindrome extends p[2*i+1]-1 characters to each side in s)
      - Even palindrome centered between s[i] and s[i+1]:  p[2*i + 2]
        is the half-length in s.

    Example:
        manacher("abacaba")
        T = "#a#b#a#c#a#b#a#"
        p = [1,2,1,4,1,2,1,8,1,2,1,4,1,2,1]
    """
    t = _transform(s)
    n = len(t)
    p = [0] * n
    c = r = 0           # center and right boundary of the rightmost palindrome
    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirror])
        # Attempt to expand
        a, b = i - (p[i] + 1), i + (p[i] + 1)
        while a >= 0 and b < n and t[a] == t[b]:
            p[i] += 1
            a -= 1
            b += 1
        if i + p[i] > r:
            c, r = i, i + p[i]
    return p


# --------------------------------------------------------------------------- #
#  Longest palindromic substring
# --------------------------------------------------------------------------- #

def longest_palindrome(s: str) -> str:
    """
    Returns the longest palindromic substring of s.
    If there are ties, returns the leftmost one.

    Example:
        longest_palindrome("babad")   -> "bab"
        longest_palindrome("cbbd")    -> "bb"
        longest_palindrome("racecar") -> "racecar"
    """
    if not s:
        return ""
    p = manacher(s)
    max_r, center = max((v, i) for i, v in enumerate(p))
    start = (center - max_r) // 2
    return s[start:start + max_r]


# --------------------------------------------------------------------------- #
#  Count all palindromic substrings (including single characters)
# --------------------------------------------------------------------------- #

def count_palindromic_substrings(s: str) -> int:
    """
    Returns the total number of palindromic substrings (single chars count).

    Key insight: each position i in T contributes floor((p[i]+1)/2) palindromes
    in the original string.

    Example:
        count_palindromic_substrings("aaa")   -> 6   # a,a,a,aa,aa,aaa
        count_palindromic_substrings("abc")   -> 3   # a,b,c
        count_palindromic_substrings("abba")  -> 6   # a,b,b,a,bb,abba
    """
    p = manacher(s)
    return sum((v + 1) // 2 for v in p)


# --------------------------------------------------------------------------- #
#  All palindrome radii for the original string
# --------------------------------------------------------------------------- #

def all_palindrome_radii(s: str) -> tuple[list[int], list[int]]:
    """
    Returns (odd_r, even_r) where:
      odd_r[i]  = radius of the longest odd-length palindrome centered at s[i].
                  The palindrome is s[i - odd_r[i] : i + odd_r[i] + 1].
      even_r[i] = radius of the longest even-length palindrome with right-center
                  at s[i] (between s[i-1] and s[i]).
                  The palindrome is s[i - even_r[i] : i + even_r[i]].

    Example:
        odd_r, even_r = all_palindrome_radii("abacaba")
        odd_r  -> [1, 1, 3, 4, 3, 1, 1]   # 4 means full string
        even_r -> [0, 0, 0, 0, 0, 0, 0]
    """
    n = len(s)
    p = manacher(s)
    odd_r = [p[2 * i + 1] for i in range(n)]       # radii at '#'-separated positions
    even_r = [p[2 * i] for i in range(n)]           # radii at original positions... wait
    # Corrected mapping:
    # T index 2i+1 corresponds to original character s[i]   -> odd center
    # T index 2i   corresponds to the '#' before s[i]       -> even center (between s[i-1] and s[i])
    odd_r = [p[2 * i + 1] for i in range(n)]        # p[2i+1] = half-len of longest odd palindrome at i
    even_r = [p[2 * i + 2] for i in range(n - 1)]   # p[2i+2] = half-len of even palindrome between i and i+1
    return odd_r, even_r


# --------------------------------------------------------------------------- #
#  Check if a substring s[l..r] (inclusive) is a palindrome in O(1) per query
# --------------------------------------------------------------------------- #

def palindrome_checker(s: str):
    """
    Returns a function is_palindrome(l, r) that answers in O(1) after O(n) build.

    Example:
        check = palindrome_checker("racecar")
        check(0, 6)  -> True
        check(0, 3)  -> False
    """
    p = manacher(s)

    def is_palindrome(l: int, r: int) -> bool:
        # Center in transformed string
        center = l + r + 1          # = (2l+1 + 2r+1) / 2  but integer form
        radius = r - l
        return p[center] >= radius

    return is_palindrome


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    assert longest_palindrome("babad") in ("bab", "aba")
    assert longest_palindrome("cbbd") == "bb"
    assert longest_palindrome("racecar") == "racecar"
    assert longest_palindrome("a") == "a"
    assert longest_palindrome("") == ""

    assert count_palindromic_substrings("aaa") == 6
    assert count_palindromic_substrings("abc") == 3
    assert count_palindromic_substrings("abba") == 6

    check = palindrome_checker("racecar")
    assert check(0, 6) is True
    assert check(0, 3) is False

    check2 = palindrome_checker("abba")
    assert check2(0, 3) is True
    assert check2(1, 2) is True
    assert check2(0, 1) is False

    print("All Manacher tests passed.")
