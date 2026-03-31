"""
Search engine for DSA templates using BM25 ranking.
Parses solution files from the solutions/ directory.
"""
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

SOLUTIONS_DIR = Path(__file__).parent / "solutions"


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
