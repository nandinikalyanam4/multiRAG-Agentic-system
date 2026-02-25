
"""
MULTI-ARMED BANDIT AGENT ROUTER
================================
Instead of asking an LLM "which agent should handle this?" every time (slow, costs money),
we use a UCB1 (Upper Confidence Bound) bandit algorithm that LEARNS which agent
works best for each question category.

HOW IT WORKS:
- Each (category, agent) pair has a track record: total attempts, total reward
- UCB1 balances exploration (trying agents we don't know much about) vs
  exploitation (using agents that have performed well)
- Over time, the router stops exploring and converges on the best agent per category

UCB1 FORMULA:
  score = average_reward + C * sqrt(ln(total_pulls) / agent_pulls)
  - First term: exploitation (pick high reward agents)
  - Second term: exploration (boost agents we haven't tried much)
  - C: exploration constant (higher = more exploration)

WHY THIS IS RL:
- State: question category
- Action: choosing an agent
- Reward: user feedback (thumbs up = 1.0, thumbs down = 0.0)
- Policy: UCB1 selection strategy
- The policy IMPROVES over time as it collects more rewards
"""
import math, json, time
from pathlib import Path
from config import settings
from threading import Lock

BANDIT_STATE_PATH = settings.base_dir / "bandit_state.json"

AGENTS = [
    "naive_rag", "sentence_window_rag", "parent_child_rag",
    "agentic_rag", "multimodal_rag", "table_rag",
    "graph_rag", "hybrid_rag", "hyde_rag", "corrective_rag",
    "lora_rag",
]

CATEGORIES = ["factual", "analytical", "relational", "procedural", "vague", "visual"]

# Smart priors — give the bandit a head start based on what we know
# Each agent gets a small bonus in categories where it should logically excel
PRIOR_REWARDS = {
    ("analytical", "table_rag"): (3, 2.4),     # 3 fake pulls, 2.4 reward → 0.8 avg
    ("relational", "graph_rag"): (3, 2.4),
    ("visual", "multimodal_rag"): (3, 2.4),
    ("vague", "hyde_rag"): (3, 2.1),            # 0.7 avg
    ("factual", "naive_rag"): (2, 1.4),         # 0.7 avg
    ("factual", "hybrid_rag"): (2, 1.4),
    ("procedural", "agentic_rag"): (3, 2.4),
    ("procedural", "parent_child_rag"): (2, 1.4),
}


class BanditRouter:
    """UCB1 Multi-Armed Bandit for agent selection."""

    def __init__(self, exploration_constant: float = 1.41):
        self.C = exploration_constant  # sqrt(2) is theoretically optimal
        self.lock = Lock()

        # State: {category: {agent: {"pulls": N, "total_reward": R}}}
        self.state = {}
        self._load()

    def _load(self):
        """Load bandit state from disk."""
        if BANDIT_STATE_PATH.exists():
            with open(BANDIT_STATE_PATH, "r") as f:
                self.state = json.load(f)
        else:
            self._initialize()

    def _save(self):
        """Persist bandit state."""
        with open(BANDIT_STATE_PATH, "w") as f:
            json.dump(self.state, f, indent=2)

    def _initialize(self):
        """Initialize with prior knowledge."""
        self.state = {}
        for cat in CATEGORIES:
            self.state[cat] = {}
            for agent in AGENTS:
                prior = PRIOR_REWARDS.get((cat, agent), (1, 0.5))  # default: 1 pull, 0.5 reward
                self.state[cat][agent] = {
                    "pulls": prior[0],
                    "total_reward": prior[1],
                }
        self._save()

    def select_agent(self, category: str) -> tuple[str, dict]:
        """
        Select the best agent for this category using UCB1.
        Returns (agent_name, debug_info with all scores).
        """
        if category not in self.state:
            category = "factual"  # fallback

        cat_state = self.state[category]
        total_pulls = sum(a["pulls"] for a in cat_state.values())

        scores = {}
        for agent, data in cat_state.items():
            n = data["pulls"]
            avg_reward = data["total_reward"] / n if n > 0 else 0

            # UCB1 formula
            exploration_bonus = self.C * math.sqrt(math.log(total_pulls + 1) / (n + 1))
            ucb_score = avg_reward + exploration_bonus

            scores[agent] = {
                "ucb_score": round(ucb_score, 4),
                "avg_reward": round(avg_reward, 4),
                "exploration_bonus": round(exploration_bonus, 4),
                "pulls": n,
                "total_reward": round(data["total_reward"], 2),
            }

        # Select agent with highest UCB score
        best_agent = max(scores, key=lambda a: scores[a]["ucb_score"])

        return best_agent, {
            "category": category,
            "selected": best_agent,
            "all_scores": scores,
            "total_interactions": total_pulls,
        }

    def update(self, category: str, agent: str, reward: float):
        """
        Update bandit after receiving feedback.
        reward: 0.0 (bad) to 1.0 (great)
        """
        if category not in self.state:
            return
        if agent not in self.state[category]:
            return

        with self.lock:
            self.state[category][agent]["pulls"] += 1
            self.state[category][agent]["total_reward"] += reward
            self._save()

    def get_stats(self) -> dict:
        """Get full bandit state with derived metrics."""
        stats = {}
        for cat in CATEGORIES:
            cat_stats = {}
            for agent in AGENTS:
                data = self.state.get(cat, {}).get(agent, {"pulls": 0, "total_reward": 0})
                n = data["pulls"]
                cat_stats[agent] = {
                    "pulls": n,
                    "avg_reward": round(data["total_reward"] / n, 3) if n > 0 else 0,
                    "total_reward": round(data["total_reward"], 2),
                }
            # Sort by avg_reward descending
            cat_stats = dict(sorted(cat_stats.items(), key=lambda x: x[1]["avg_reward"], reverse=True))
            stats[cat] = cat_stats
        return stats

    def get_leaderboard(self) -> list[dict]:
        """Ranked list of agents with overall performance."""
        agent_totals = {a: {"pulls": 0, "reward": 0, "categories_won": 0} for a in AGENTS}

        for cat in CATEGORIES:
            best_avg = 0
            best_agent = None
            for agent in AGENTS:
                data = self.state.get(cat, {}).get(agent, {"pulls": 0, "total_reward": 0})
                agent_totals[agent]["pulls"] += data["pulls"]
                agent_totals[agent]["reward"] += data["total_reward"]
                avg = data["total_reward"] / data["pulls"] if data["pulls"] > 0 else 0
                if avg > best_avg:
                    best_avg = avg
                    best_agent = agent
            if best_agent:
                agent_totals[best_agent]["categories_won"] += 1

        leaderboard = []
        for agent, totals in agent_totals.items():
            n = totals["pulls"]
            leaderboard.append({
                "agent": agent,
                "total_interactions": n,
                "avg_reward": round(totals["reward"] / n, 3) if n > 0 else 0,
                "categories_won": totals["categories_won"],
            })

        leaderboard.sort(key=lambda x: x["avg_reward"], reverse=True)
        return leaderboard

    def reset(self):
        """Reset bandit to initial priors."""
        self._initialize()