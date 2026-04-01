"""
Search engine for DSA templates using BM25 ranking.
Parses solution files from the solutions/ directory.
"""
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

SOLUTIONS_DIR = Path(__file__).parent / "solutions"

# Maps keyword patterns (regex) → (signal label, list of template name fragments to boost, expected category)
SIGNALS: list[tuple[str, str, list[str], str]] = [
    # Graphs
    (r"\bshortest path\b|\bmin(?:imum)? dist",       "Shortest path problem",          ["dijkstra", "bellman", "floyd", "bfs"], "graphs"),
    (r"\bnegative (?:weight|edge|cycle)",             "Negative weights detected",      ["bellman", "floyd"], "graphs"),
    (r"\ball.pairs",                                   "All-pairs shortest path",        ["floyd"], "graphs"),
    (r"\bminimum spanning tree\b|\bmst\b",            "MST problem",                    ["spanning tree"], "graphs"),
    (r"\bconnected component",                        "Connected components",            ["dfs", "bfs", "dsu"], "graphs"),
    (r"\bcycle\b",                                    "Cycle detection",                ["cycle", "dsu", "dfs"], "graphs"),
    (r"\btopolog",                                    "Topological ordering",           ["topological"], "graphs"),
    (r"\bstrongly connected\b|\bscc\b",               "Strongly connected components",  ["tarjan", "kosaraju", "scc"], "graphs"),
    (r"\bbipartite\b|\b2.colou?r",                    "Bipartite / 2-coloring",         ["bipartite"], "graphs"),
    (r"\bmax(?:imum)? flow\b|\bmin cut\b",            "Network flow",                   ["flow"], "graphs"),
    (r"\bgrid\b|\bmatrix\b.*\bpath\b|\bmaze\b",       "Grid traversal",                 ["grid", "bfs"], "graphs"),
    # Trees
    (r"\brange (?:sum|min|max|query)\b",              "Range query",                    ["segment tree", "fenwick", "sparse"], "trees"),
    (r"\brange update\b|\blazy",                      "Range update",                   ["lazy"], "trees"),
    (r"\bprefix sum\b|\bpoint update",                "Prefix sum / point update",      ["fenwick", "bit"], "trees"),
    (r"\blca\b|\blowest common ancestor",             "LCA query",                      ["lca"], "trees"),
    (r"\bsubtree\b|\btree dp\b",                      "Tree DP",                        ["tree dp", "dp on tree"], "trees"),
    (r"\btrie\b|\bprefix\b.*\bstring",                "Prefix / trie",                  ["trie"], "trees"),
    (r"\bxor\b.*\bmax\b|\bmax\b.*\bxor",             "XOR maximization",               ["trie"], "trees"),
    (r"\bunion.find\b|\bdisjoint\b|\bdsu\b",          "Union-Find / DSU",               ["dsu", "union"], "data_structures"),
    # DP
    (r"\bknapsack\b|\bweight\b.*\bvalue\b|\bpack",    "Knapsack-style DP",              ["knapsack"], "dp"),
    (r"\blongest (?:common )?subsequence\b|\blcs\b",  "LCS",                            ["lcs"], "dp"),
    (r"\blongest increasing\b|\blis\b",               "LIS",                            ["lis"], "dp"),
    (r"\bedit distance\b|\blevenshtein",              "Edit distance",                  ["edit distance"], "dp"),
    (r"\bcoin\b|\bchange\b.*\bmin",                   "Coin change",                    ["coin change"], "dp"),
    (r"\bbitmask\b|\bsubset\b.*\bdp\b|\btsp\b",       "Bitmask DP",                     ["bitmask"], "dp"),
    (r"\bdigit dp\b|\bcount.*\bdigit\b",              "Digit DP",                       ["digit dp"], "dp"),
    (r"\binterval\b.*\bdp\b|\bburst\b|\bmerge.*stone","Interval DP",                    ["interval dp"], "dp"),
    (r"\bexpect(?:ed)?\b.*\bvalue\b|\bprobabilit",    "Probability / expected value DP",["probability"], "dp"),
    # Strings
    (r"\bpattern match\b|\boccurrences? of\b",        "Pattern matching",               ["kmp", "z-algo", "rabin"], "strings"),
    (r"\bmulti.pattern\b|\bmany patterns\b",          "Multi-pattern matching",         ["aho-corasick"], "strings"),
    (r"\bpalindrome\b",                               "Palindrome",                     ["manacher", "palindrome"], "strings"),
    (r"\bsuffix\b|\bsubstring.*distinct\b",           "Suffix structure",               ["suffix array"], "strings"),
    (r"\brolling hash\b|\bstring hash",               "String hashing",                 ["hashing"], "strings"),
    # Math
    (r"\bprime\b|\bsieve\b|\bprimality",              "Prime numbers",                  ["sieve", "prime"], "math_nt"),
    (r"\bmod(?:ular)?\b.*\binverse\b|\bmodulo\b",     "Modular arithmetic",             ["modular"], "math_nt"),
    (r"\bcombination\b|\bbinomial\b|\bncr\b",         "Combinatorics",                  ["combinatorics"], "math_nt"),
    (r"\bchinese remainder\b|\bcrt\b",                "CRT",                            ["chinese remainder"], "math_nt"),
    (r"\bmatrix.*(?:power|exp)\b|\brecurrence",       "Matrix exponentiation",          ["matrix exp"], "math_nt"),
    (r"\bfft\b|\bpolynomial\b|\bconvolution",         "FFT / polynomial",               ["fft"], "math_nt"),
    (r"\bconvex hull\b|\bgeometr",                    "Geometry",                       ["geometry", "convex hull"], "math_nt"),
    # Data structures
    (r"\bnext greater\b|\bnext smaller\b|\bhistogram","Monotonic stack",                ["monotonic stack"], "data_structures"),
    (r"\bsliding window\b|\bwindow.*max\b",           "Sliding window",                 ["sliding window", "monotonic deque"], "data_structures"),
    (r"\bk.?th\b|\btop.k\b|\bmedian",                "Order statistics / heap",        ["heap", "ordered set"], "data_structures"),
    (r"\binversion\b|\bcount.*smaller",               "Inversion count",                ["merge sort", "bit", "fenwick"], "data_structures"),
    (r"\bbinary search.*answer\b|\bminimize.*max\b",  "Binary search on answer",        ["binary search"], "data_structures"),
    (r"\bsqrt\b.*\bdecompos\b|\bmo.?s\b",            "Mo's / sqrt decomposition",      ["sqrt", "mo"], "data_structures"),
    # Ad-hoc / simulation patterns
    (r"\bsort\b|\bordering\b|\barrang",               "Sorting / ordering",             ["sort"], "data_structures"),
    (r"\bsimulat",                                    "Simulation",                     ["simulation"], "data_structures"),
    (r"\bgreedy\b",                                   "Greedy algorithm",               ["greedy"], "data_structures"),
    (r"\bimplementation\b|\badhoc\b",                 "Ad-hoc / implementation",        ["implementation"], "data_structures"),
    # Array patterns
    (r"\bsubarray\b",                                 "Subarray problem",               ["subarray"], "data_structures"),
    (r"\bsubsequence\b",                              "Subsequence problem",            ["subsequence"], "dp"),
    (r"\brotation\b|\brotate\b",                      "Array rotation",                 ["rotation"], "data_structures"),
    (r"\bpermutation\b",                              "Permutation",                    ["permutation"], "data_structures"),
    # Competition-specific
    (r"\btest case",                                  "Multiple test cases",            [], "data_structures"),
    (r"\bconstraint",                                 "Constraint analysis",            [], "data_structures"),
]


def _tokenize(text: str) -> list:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _parse_solution_file(filepath: Path) -> dict | None:
    try:
        content = filepath.read_text(encoding="utf-8")
        category = filepath.parent.name

        meta = {
            "name": filepath.stem.replace("_", " ").title(),
            "tags": [],
            "description": "",
            "complexity": "",
            "code": content,
            "file": str(filepath),
            "category": category,
        }

        doc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if doc_match:
            docstring = doc_match.group(1)

            m = re.search(r"NAME:\s*(.+)", docstring)
            if m:
                meta["name"] = m.group(1).strip()

            m = re.search(r"TAGS:\s*(.+)", docstring)
            if m:
                meta["tags"] = [t.strip() for t in m.group(1).split(",")]

            m = re.search(r"DESCRIPTION:\s*(.+?)(?:COMPLEXITY:|$)", docstring, re.DOTALL)
            if m:
                meta["description"] = re.sub(r"\s+", " ", m.group(1)).strip()

            m = re.search(r"COMPLEXITY:\s*(.+)", docstring)
            if m:
                meta["complexity"] = m.group(1).strip()

        if category.replace("_", "-") not in meta["tags"]:
            meta["tags"].append(category.replace("_", "-"))

        return meta
    except Exception as e:
        print(f"Warning: could not parse {filepath}: {e}")
        return None


class SearchEngine:
    def __init__(self):
        self.solutions: list[dict] = []
        self._corpus: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 1.0
        self._load()
        self._build_index()

    def reload(self):
        self.solutions = []
        self._load()
        self._build_index()

    def _load(self):
        if not SOLUTIONS_DIR.exists():
            SOLUTIONS_DIR.mkdir(parents=True)
            return
        for fp in sorted(SOLUTIONS_DIR.rglob("*.py")):
            if fp.name.startswith("_"):
                continue
            sol = _parse_solution_file(fp)
            if sol:
                self.solutions.append(sol)

    def _build_index(self):
        self._corpus = []
        for sol in self.solutions:
            tokens = (
                _tokenize(sol["name"]) * 4
                + _tokenize(" ".join(sol["tags"])) * 3
                + _tokenize(sol["description"]) * 2
                + _tokenize(sol.get("code", ""))[:150]
            )
            self._corpus.append(tokens)

        N = len(self._corpus)
        if N == 0:
            return

        df: dict[str, int] = defaultdict(int)
        for doc in self._corpus:
            for term in set(doc):
                df[term] += 1

        self._idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }
        self._avgdl = sum(len(d) for d in self._corpus) / N

    def _bm25(self, query_terms: list, doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        doc = self._corpus[doc_idx]
        dl = len(doc)
        tf = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = self._idf.get(term, 0.0)
            tf_v = tf[term]
            score += idf * (tf_v * (k1 + 1)) / (tf_v + k1 * (1 - b + b * dl / self._avgdl))
        return score

    def search(self, query: str, top_k: int = 30) -> list[dict]:
        if not self.solutions:
            return []
        q_terms = _tokenize(query)
        scored = []
        for i, sol in enumerate(self.solutions):
            score = self._bm25(q_terms, i)
            # Tag exact match boost
            sol_tags = {t.lower().replace("-", "").replace("_", "") for t in sol["tags"]}
            for term in q_terms:
                if term in sol_tags:
                    score += 6.0
            # Name token boost
            for term in q_terms:
                if term in _tokenize(sol["name"]):
                    score += 4.0
            scored.append((i, score))

        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self.solutions[i] for i, _ in scored[:top_k]]

    def get_all(self) -> list[dict]:
        return sorted(self.solutions, key=lambda s: (s["category"], s["name"]))

    def classify(self, problem_text: str, top_k: int = 10) -> dict:
        """
        Classify a problem description and return structured results.

        Returns:
            {
                "primary_category": str,  # Most likely category (graphs, dp, etc.)
                "detected_patterns": [(pattern_label, confidence), ...],
                "ranked_solutions": [solution_dict, ...],
                "confidence": float  # Overall classification confidence (0-100)
            }
        """
        if not problem_text.strip():
            return {
                "primary_category": "unknown",
                "detected_patterns": [],
                "ranked_solutions": [],
                "confidence": 0.0,
            }

        problem_lower = problem_text.lower()

        # Step 1: Pattern matching to detect signals
        triggered: list[tuple[str, list[str], str]] = []  # (label, boost_fragments, expected_category)
        for pattern, label, boosts, category in SIGNALS:
            if re.search(pattern, problem_lower):
                triggered.append((label, boosts, category))

        # Step 2: BM25 retrieval
        retrieved = self.search(problem_text, top_k=top_k)

        # Step 3: Signal boosting
        boost_map: dict[str, float] = {}
        for sol in retrieved:
            name_lower = sol["name"].lower()
            tags_lower = " ".join(sol["tags"]).lower()
            text = name_lower + " " + tags_lower
            extra = 0.0
            for _, frags, _ in triggered:
                for frag in frags:
                    if frag in text:
                        extra += 6.0
                        break
            boost_map[sol["name"]] = extra

        # Re-rank by BM25 position + boost
        scored = sorted(
            enumerate(retrieved),
            key=lambda x: boost_map.get(x[1]["name"], 0) - x[0] * 0.5,
            reverse=True,
        )
        reranked = [sol for _, sol in scored]

        # Step 4: Determine primary category
        primary_category, confidence = self._determine_primary_category(
            reranked[:top_k], triggered
        )

        # Step 5: Build detected patterns with confidence
        detected_patterns = [
            (label, self._calculate_pattern_confidence(label, boosts, reranked[:6]))
            for label, boosts, _ in triggered
        ]
        detected_patterns.sort(key=lambda x: x[1], reverse=True)

        return {
            "primary_category": primary_category,
            "detected_patterns": detected_patterns,
            "ranked_solutions": reranked[:6],
            "confidence": confidence,
        }

    def _determine_primary_category(
        self,
        solutions: list[dict],
        triggered_signals: list[tuple[str, list[str], str]]
    ) -> tuple[str, float]:
        """
        Determine primary category from ranked solutions and signals.
        Returns (category_name, confidence_score).
        """
        if not solutions:
            return ("unknown", 0.0)

        # Count categories in top solutions (weighted by position)
        category_scores: dict[str, float] = defaultdict(float)
        for i, sol in enumerate(solutions[:6]):
            weight = 1.0 / (i + 1)  # First result gets weight 1.0, second 0.5, etc.
            category_scores[sol["category"]] += weight * 10.0

        # Add signal-based category votes
        signal_categories = Counter(cat for _, _, cat in triggered_signals)
        for cat, count in signal_categories.items():
            category_scores[cat] += count * 5.0

        if not category_scores:
            return ("unknown", 0.0)

        # Find category with highest score
        primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
        max_score = category_scores[primary_category]

        # Calculate confidence based on:
        # - Number of patterns matched (more patterns = higher confidence)
        # - Score margin (how much better is the top category vs others)
        # - Consistency of top results
        pattern_factor = min(len(triggered_signals) * 15, 40)  # Up to 40% from patterns

        sorted_scores = sorted(category_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
            margin_factor = margin * 30  # Up to 30% from margin
        else:
            margin_factor = 30  # Full margin bonus if only one category

        # Check consistency: do top 3 results agree on category?
        top3_categories = [sol["category"] for sol in solutions[:min(3, len(solutions))]]
        consistency = top3_categories.count(primary_category) / len(top3_categories)
        consistency_factor = consistency * 30  # Up to 30% from consistency

        confidence = min(pattern_factor + margin_factor + consistency_factor, 100.0)

        return (primary_category, round(confidence, 1))

    def _calculate_pattern_confidence(
        self, label: str, boost_fragments: list[str], top_solutions: list[dict]
    ) -> float:
        """
        Calculate confidence for a detected pattern based on how many top solutions match.
        Returns confidence score (0-100).
        """
        if not top_solutions:
            return 50.0  # Pattern detected but no solutions to validate

        matches = 0
        for sol in top_solutions:
            name_lower = sol["name"].lower()
            tags_lower = " ".join(sol["tags"]).lower()
            text = name_lower + " " + tags_lower
            for frag in boost_fragments:
                if frag in text:
                    matches += 1
                    break

        # Confidence based on match rate in top solutions
        match_rate = matches / len(top_solutions)
        base_confidence = 50.0 + (match_rate * 50.0)  # 50-100 range

        return round(base_confidence, 1)
