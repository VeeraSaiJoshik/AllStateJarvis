"""
NAME: Aho-Corasick Multi-Pattern String Matching
TAGS: string, trie, automaton, multi-pattern, failure links, linear time
DESCRIPTION: Aho-Corasick builds a trie from a dictionary of patterns and augments it
             with failure (suffix) links and dictionary links, enabling simultaneous
             search for all patterns in O(n + total_matches) time after O(sum_m) build.
             The go-to algorithm for problems like virus scanning, word filtering,
             or finding which of many keywords appear in a text.
COMPLEXITY: Time O(Σ|patterns| + n + matches), Space O(Σ|patterns| * ALPHA)
"""

from collections import deque

ALPHA = 26      # change to 128 for full ASCII


def _ord(c: str) -> int:
    return ord(c) - ord('a')


# --------------------------------------------------------------------------- #
#  Aho-Corasick Automaton
# --------------------------------------------------------------------------- #

class AhoCorasick:
    """
    Usage:
        ac = AhoCorasick()
        ac.add_pattern("he")
        ac.add_pattern("she")
        ac.add_pattern("his")
        ac.add_pattern("hers")
        ac.build()
        results = ac.search("ushers")
        # results: [("he", 1), ("she", 1), ("hers", 1)]  (pattern, start_index)
    """

    def __init__(self):
        # Each node: [children (array of size ALPHA), fail, output]
        # output: list of pattern indices that end at this node
        self.goto = [[- 1] * ALPHA]     # goto[node][char] = next node
        self.fail = [0]
        self.output: list[list[int]] = [[]]
        self.patterns: list[str] = []

    def add_pattern(self, pattern: str) -> int:
        """Inserts pattern into trie. Returns pattern index."""
        node = 0
        for c in pattern:
            ch = _ord(c)
            if self.goto[node][ch] == -1:
                self.goto[node][ch] = len(self.goto)
                self.goto.append([-1] * ALPHA)
                self.fail.append(0)
                self.output.append([])
            node = self.goto[node][ch]
        idx = len(self.patterns)
        self.patterns.append(pattern)
        self.output[node].append(idx)
        return idx

    def build(self):
        """Compute failure links and propagate outputs using BFS."""
        q = deque()
        # Initialize depth-1 nodes
        for ch in range(ALPHA):
            nxt = self.goto[0][ch]
            if nxt == -1:
                self.goto[0][ch] = 0    # loop back to root for missing chars
            else:
                self.fail[nxt] = 0
                q.append(nxt)

        while q:
            node = q.popleft()
            for ch in range(ALPHA):
                nxt = self.goto[node][ch]
                if nxt == -1:
                    # Shortcut: reuse failure-link's transition
                    self.goto[node][ch] = self.goto[self.fail[node]][ch]
                else:
                    self.fail[nxt] = self.goto[self.fail[node]][ch]
                    # Merge outputs via failure chain
                    self.output[nxt] += self.output[self.fail[nxt]]
                    q.append(nxt)

    def search(self, text: str) -> list[tuple[str, int]]:
        """
        Returns list of (pattern, start_index) for every match in text.
        Matches are reported in the order they end in the text.

        Example:
            ac.search("ushers") -> [("he", 2), ("she", 1), ("hers", 2)]
        """
        node = 0
        results = []
        for i, c in enumerate(text):
            ch = _ord(c)
            node = self.goto[node][ch]
            for pat_idx in self.output[node]:
                pat = self.patterns[pat_idx]
                start = i - len(pat) + 1
                results.append((pat, start))
        return results

    def search_count(self, text: str) -> dict[str, int]:
        """
        Returns a dict mapping each pattern to its occurrence count in text.

        Example:
            ac.search_count("ushers") -> {"he": 1, "she": 1, "hers": 1, "his": 0}
        """
        count = {p: 0 for p in self.patterns}
        node = 0
        for c in text:
            node = self.goto[node][_ord(c)]
            for pat_idx in self.output[node]:
                count[self.patterns[pat_idx]] += 1
        return count


# --------------------------------------------------------------------------- #
#  Convenience function
# --------------------------------------------------------------------------- #

def aho_corasick_search(text: str, patterns: list[str]) -> list[tuple[str, int]]:
    """
    One-shot: build automaton and return all (pattern, start) matches.

    Example:
        aho_corasick_search("ushers", ["he", "she", "his", "hers"])
        -> [("she", 1), ("he", 2), ("hers", 2)]
    """
    ac = AhoCorasick()
    for p in patterns:
        ac.add_pattern(p)
    ac.build()
    return ac.search(text)


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ac = AhoCorasick()
    for p in ["he", "she", "his", "hers"]:
        ac.add_pattern(p)
    ac.build()

    results = ac.search("ushers")
    result_set = set((pat, start) for pat, start in results)
    assert ("he", 2) in result_set, result_set
    assert ("she", 1) in result_set, result_set
    assert ("hers", 2) in result_set, result_set

    counts = ac.search_count("ushers")
    assert counts["he"] == 1
    assert counts["she"] == 1
    assert counts["his"] == 0
    assert counts["hers"] == 1

    # Multiple occurrences
    ac2 = AhoCorasick()
    ac2.add_pattern("ab")
    ac2.add_pattern("bc")
    ac2.build()
    res2 = ac2.search("ababc")
    pairs = [(p, s) for p, s in res2]
    assert ("ab", 0) in pairs
    assert ("ab", 2) in pairs
    assert ("bc", 3) in pairs

    print("All Aho-Corasick tests passed.")
