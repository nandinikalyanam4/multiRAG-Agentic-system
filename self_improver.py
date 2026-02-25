
"""
SELF-IMPROVING RETRIEVAL
=========================
Analyzes accumulated interaction logs to find patterns and suggest/apply
parameter adjustments.

What it looks at:
- Which agents consistently get low rewards → suggest deprecating or fixing
- Which categories have low overall satisfaction → suggest adding training data
- Correlation between top_k and reward → adjust default top_k
- Time-of-day patterns, query length patterns, etc.
"""
from reward_store import load_all_interactions
from bandit_router import BanditRouter, AGENTS, CATEGORIES

class SelfImprover:
    def __init__(self, bandit: BanditRouter):
        self.bandit = bandit

    def analyze(self) -> dict:
        """Run full analysis on accumulated data."""
        interactions = load_all_interactions()

        if len(interactions) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 10 interactions with feedback. Currently have {len(interactions)}.",
                "interactions_logged": len(interactions),
            }

        report = {
            "total_interactions": len(interactions),
            "with_feedback": len([i for i in interactions if "reward" in i]),
            "insights": [],
            "recommendations": [],
            "parameter_suggestions": {},
        }

        # ---- Analysis 1: Agent performance by category ----
        agent_cat_perf = {}
        for i in interactions:
            if "reward" not in i:
                continue
            key = (i.get("agent", "unknown"), i.get("category", "unknown"))
            if key not in agent_cat_perf:
                agent_cat_perf[key] = []
            agent_cat_perf[key].append(i["reward"])

        # Find struggling agents
        for (agent, cat), rewards in agent_cat_perf.items():
            avg = sum(rewards) / len(rewards)
            if avg < 0.4 and len(rewards) >= 3:
                report["insights"].append({
                    "type": "low_performance",
                    "agent": agent,
                    "category": cat,
                    "avg_reward": round(avg, 3),
                    "sample_size": len(rewards),
                    "message": f"{agent} struggles with {cat} questions (avg reward: {avg:.2f})"
                })

        # Find star agents
        for (agent, cat), rewards in agent_cat_perf.items():
            avg = sum(rewards) / len(rewards)
            if avg > 0.8 and len(rewards) >= 3:
                report["insights"].append({
                    "type": "high_performance",
                    "agent": agent,
                    "category": cat,
                    "avg_reward": round(avg, 3),
                    "sample_size": len(rewards),
                    "message": f"{agent} excels at {cat} questions (avg reward: {avg:.2f})"
                })

        # ---- Analysis 2: Top-K effectiveness ----
        topk_rewards = {}
        for i in interactions:
            if "reward" not in i or "top_k" not in i:
                continue
            k = i["top_k"]
            if k not in topk_rewards:
                topk_rewards[k] = []
            topk_rewards[k].append(i["reward"])

        if topk_rewards:
            best_k = max(topk_rewards, key=lambda k: sum(topk_rewards[k]) / len(topk_rewards[k]))
            report["parameter_suggestions"]["recommended_top_k"] = best_k
            report["insights"].append({
                "type": "top_k_analysis",
                "best_k": best_k,
                "all_k_performance": {
                    k: {"avg_reward": round(sum(v)/len(v), 3), "count": len(v)}
                    for k, v in topk_rewards.items()
                },
            })

        # ---- Analysis 3: Category difficulty ----
        cat_rewards = {}
        for i in interactions:
            if "reward" not in i:
                continue
            cat = i.get("category", "unknown")
            if cat not in cat_rewards:
                cat_rewards[cat] = []
            cat_rewards[cat].append(i["reward"])

        hardest = None
        hardest_avg = 1.0
        for cat, rewards in cat_rewards.items():
            avg = sum(rewards) / len(rewards)
            if avg < hardest_avg:
                hardest_avg = avg
                hardest = cat

        if hardest:
            report["insights"].append({
                "type": "hardest_category",
                "category": hardest,
                "avg_reward": round(hardest_avg, 3),
                "message": f"'{hardest}' questions are hardest (avg reward: {hardest_avg:.2f}). Consider uploading more relevant documents.",
            })

        # ---- Analysis 4: Query length vs reward ----
        short_q = [i for i in interactions if "reward" in i and len(i.get("question", "").split()) <= 5]
        long_q = [i for i in interactions if "reward" in i and len(i.get("question", "").split()) > 10]

        if short_q and long_q:
            short_avg = sum(i["reward"] for i in short_q) / len(short_q)
            long_avg = sum(i["reward"] for i in long_q) / len(long_q)
            report["insights"].append({
                "type": "query_length",
                "short_queries_avg": round(short_avg, 3),
                "long_queries_avg": round(long_avg, 3),
                "message": f"Short queries avg reward: {short_avg:.2f}, Long queries: {long_avg:.2f}",
            })

        # ---- Generate recommendations ----
        stats = self.bandit.get_stats()
        for cat, agents in stats.items():
            best = max(agents, key=lambda a: agents[a]["avg_reward"])
            worst = min(agents, key=lambda a: agents[a]["avg_reward"])
            if agents[best]["avg_reward"] - agents[worst]["avg_reward"] > 0.3:
                report["recommendations"].append(
                    f"For '{cat}' questions, strongly prefer {best} "
                    f"(avg: {agents[best]['avg_reward']}) over {worst} "
                    f"(avg: {agents[worst]['avg_reward']})"
                )

        return report