
"""
Classifies questions into categories WITHOUT using LLM calls.
This is fast and free — the bandit learns per category.

Categories:
  - factual: "what is", "who is", "when did", "define"
  - analytical: "compare", "analyze", "how many", "average", "trend"
  - relational: "related to", "connected", "between", "relationship"
  - procedural: "how to", "steps", "process", "explain how"
  - vague: short queries, "tell me about", no clear intent
  - visual: "image", "diagram", "picture", "chart", "screenshot"
"""
import re

CATEGORY_PATTERNS = {
    "factual": [
        r"\b(what is|who is|when did|where is|define|what does|what are)\b",
        r"\b(name of|tell me the|which one)\b",
    ],
    "analytical": [
        r"\b(compare|analyze|how many|average|total|sum|count|trend|percentage|ratio)\b",
        r"\b(statistics|data|numbers|grouped by|breakdown)\b",
    ],
    "relational": [
        r"\b(relat|connect|between|linked|associated|works with|reports to)\b",
        r"\b(hierarchy|network|chain|depend)\b",
    ],
    "procedural": [
        r"\b(how to|how do|steps to|process of|explain how|guide|tutorial)\b",
        r"\b(implement|build|create|set up|configure)\b",
    ],
    "visual": [
        r"\b(image|picture|diagram|chart|screenshot|photo|figure|graph|plot)\b",
    ],
}


def classify_question(question: str) -> str:
    """Classify question into a category using pattern matching. Fast, free, no LLM."""
    q = question.lower().strip()

    # Check each category
    scores = {}
    for category, patterns in CATEGORY_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, q))
        if score > 0:
            scores[category] = score

    if not scores:
        # Short or vague queries
        if len(q.split()) <= 4:
            return "vague"
        return "factual"  # default

    return max(scores, key=scores.get)