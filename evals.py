import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from rl_orchestrator import RLOrchestrator
from llm import llm_call


EVAL_DIR = Path("data/eval")
EVAL_QUERIES_PATH = EVAL_DIR / "queries.jsonl"


def load_eval_queries(max_questions: int | None = None) -> List[Dict[str, Any]]:
    """
    Load eval queries from data/eval/queries.jsonl if present.
    Fallback to a small built-in demo set so the script always runs.
    """
    if EVAL_QUERIES_PATH.exists():
        queries: List[Dict[str, Any]] = []
        with open(EVAL_QUERIES_PATH, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                queries.append(json.loads(line))
        if max_questions is not None:
            queries = queries[:max_questions]
        return queries

    demo = [
        {
            "id": "demo_1",
            "question": "Give a concise summary of the main topics covered in the uploaded documents.",
        },
        {
            "id": "demo_2",
            "question": "List three key risks mentioned in the documents and explain why they matter.",
        },
    ]
    if max_questions is not None:
        demo = demo[:max_questions]
    return demo


async def run_agents(
    orchestrator: RLOrchestrator,
    question: str,
    agents: List[str],
    top_k: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple agents on the same question and return their raw results.
    Uses RLOrchestrator.compare to keep behavior consistent with the API.
    """
    result = await orchestrator.compare(question, agents, top_k=top_k)
    return result.get("comparisons", {})


def judge_with_llm(question: str, answers: Dict[str, str]) -> Dict[str, Any]:
    """
    Use the primary LLM as a judge to score each agent's answer.

    Returns structure:
      {
        "scores": {agent: float 0-1},
        "winner": "<agent-name>",
        "explanation": "natural language rationale"
      }
    """
    system = (
        "You are an impartial evaluator for a retrieval-augmented QA system. "
        "You will be given a user question and multiple agent answers. "
        "Score each answer between 0.0 (useless / wrong) and 1.0 (fully correct, concise, well grounded). "
        "Prefer answers that are specific, grounded, and honest about uncertainty."
    )

    answers_block = "\n\n".join(
        [f"[{agent}]\n{text}" for agent, text in answers.items()]
    )

    user = f"""
Question:
{question}

Agent answers:
{answers_block}

Respond in JSON with the following schema:
{{
  "scores": {{"<agent>": 0.0-1.0, ...}},
  "winner": "<agent-with-highest-score>",
  "explanation": "brief natural language explanation"
}}
"""
    raw = llm_call(system, user, json_mode=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: if something went wrong, assign equal scores
        return {
            "scores": {a: 0.5 for a in answers.keys()},
            "winner": sorted(answers.keys())[0],
            "explanation": "Fallback scoring: JSON parsing failed.",
        }
    return parsed


async def run_eval(
    agents: List[str],
    top_k: int,
    max_questions: int | None,
    judge: str,
) -> Dict[str, Any]:
    """
    Core offline evaluation loop.
    """
    orchestrator = RLOrchestrator()
    queries = load_eval_queries(max_questions)

    results: List[Dict[str, Any]] = []

    for q in queries:
        qid = q.get("id") or q.get("name") or q.get("question")[:32]
        question = q["question"]

        comps = await run_agents(orchestrator, question, agents, top_k)

        # Extract plain-text answers (best-effort: fall back to str(result))
        answers: Dict[str, str] = {}
        for name, payload in comps.items():
            if isinstance(payload, dict):
                text = (
                    payload.get("answer")
                    or payload.get("response")
                    or payload.get("output")
                    or payload.get("content")
                    or json.dumps(payload)
                )
            else:
                text = str(payload)
            answers[name] = text

        if judge == "none":
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "answers": answers,
                }
            )
        else:
            judged = judge_with_llm(question, answers)
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "answers": answers,
                    "judgement": judged,
                }
            )

    # Aggregate simple statistics
    summary: Dict[str, Any] = {
        "agents": agents,
        "top_k": top_k,
        "num_questions": len(results),
        "per_agent": {},
    }

    if judge != "none":
        per_agent_scores: Dict[str, List[float]] = {a: [] for a in agents}
        win_counts: Dict[str, int] = {a: 0 for a in agents}

        for r in results:
            j = r.get("judgement") or {}
            scores = j.get("scores") or {}
            for a, s in scores.items():
                if a in per_agent_scores:
                    per_agent_scores[a].append(float(s))
            winner = j.get("winner")
            if winner in win_counts:
                win_counts[winner] += 1

        for a in agents:
            s = per_agent_scores[a]
            avg = sum(s) / len(s) if s else 0.0
            summary["per_agent"][a] = {
                "avg_score": round(avg, 3),
                "num_scored": len(s),
                "wins": win_counts.get(a, 0),
            }

    return {
        "summary": summary,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluation / A-B testing harness for multi-agent RAG."
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="naive_rag,agentic_rag,hybrid_rag",
        help="Comma-separated list of agent names to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved documents per agent.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum number of eval questions to run.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        choices=["none", "llm"],
        default="none",
        help="Evaluation mode: 'none' for raw outputs only, 'llm' to use LLM-as-a-judge.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/results.json",
        help="Where to write eval results (JSON).",
    )

    args = parser.parse_args()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        run_eval(
            agents=agents,
            top_k=args.top_k,
            max_questions=args.max_questions,
            judge=args.judge,
        )
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=== EVAL SUMMARY ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull results written to: {output_path}")


if __name__ == "__main__":
    main()

