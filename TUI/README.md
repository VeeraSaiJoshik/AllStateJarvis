# DSA Vault

A terminal-based competitive programming tool for searching algorithm templates and getting intelligent problem analysis recommendations.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Textual](https://img.shields.io/badge/TUI-Textual-green)

## Overview

DSA Vault gives you instant access to 67 curated algorithm implementations directly in your terminal. Type a keyword, preview the code, copy it to your clipboard, and get back to solving.

It also includes a problem **Advisor** tab — paste a problem statement and it detects algorithmic patterns, telling you which templates to reach for and why.

## Features

- **BM25 search engine** with name/tag/description boosting for relevant results
- **67 algorithm templates** across 6 categories
- **Advisor tab** with 84 signal patterns for problem analysis
- **Vim-style navigation** (j/k keys)
- **Copy to clipboard** with Ctrl+Y
- GitHub-styled dark theme

## Installation

```bash
pip install -r requirements.txt
python main.py
```

## Usage

### Search Tab
Type any keyword to find relevant templates. Navigate results with `j`/`k` or arrow keys, and preview code on the right.

| Keybinding | Action |
|---|---|
| `j` / `k` | Navigate results |
| `Ctrl+Y` | Copy code to clipboard |
| `Ctrl+R` | Reload templates |
| `Ctrl+Q` | Quit |

### Advisor Tab
Paste a competitive programming problem statement and press `Ctrl+Enter`. The advisor detects signals like "shortest path", "knapsack", or "palindrome" and recommends the top 6 matching templates with reasoning.

## Template Library

| Category | Count | Examples |
|---|---|---|
| Graphs | 12 | Dijkstra, Bellman-Ford, SCC, MST |
| Trees | 10 | Segment tree, Fenwick tree, DSU, LCA |
| Dynamic Programming | 12 | Knapsack, LIS, Bitmask DP, Digit DP |
| Data Structures | 12 | Binary search, Monotonic stack, Heap |
| Strings | 10 | KMP, Z-algo, Aho-Corasick, Manacher |
| Math & NT | 11 | FFT, CRT, Matrix exponentiation, Sieve |

## Adding Templates

Create a `.py` file in the appropriate `solutions/<category>/` subdirectory using this docstring format:

```python
"""
NAME: Your Algorithm Name
TAGS: tag1, tag2, tag3
DESCRIPTION: When and why to use this algorithm.
COMPLEXITY: Time: O(...), Space: O(...)
CODE:
"""

# your implementation here
```

The search engine picks it up automatically on the next load or `Ctrl+R`.
