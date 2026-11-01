"""Discussion engine: runs a configurable flow of ChatGPT/Claude steps
until a reviewer's score crosses the threshold (or the round cap is hit).

A flow is two lists of steps:
  - opening: run once, in order (e.g. ChatGPT answers, Claude improves)
  - loop:    repeated in order until the stop condition is met
             (e.g. ChatGPT reviews, Claude revises)

Each step is (agent, action), where agent is "ChatGPT" or "Claude" and
action is one of:
  - "Answer":  write an answer to the question from scratch
  - "Improve": rewrite the current answer, fixing any issues raised
  - "Review":  critique the current answer and give it a 0-10 score
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

ACTIONS = ["Answer", "Improve", "Review"]
AGENTS = ["ChatGPT", "Claude"]

PRESETS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "ChatGPT answers → Claude improves → ChatGPT validates ↺": {
        "opening": [("ChatGPT", "Answer"), ("Claude", "Improve")],
        "loop": [("ChatGPT", "Review"), ("Claude", "Improve")],
    },
    "Claude answers → ChatGPT improves → Claude validates ↺": {
        "opening": [("Claude", "Answer"), ("ChatGPT", "Improve")],
        "loop": [("Claude", "Review"), ("ChatGPT", "Improve")],
    },
    "ChatGPT answers → Claude reviews → ChatGPT revises ↺": {
        "opening": [("ChatGPT", "Answer")],
        "loop": [("Claude", "Review"), ("ChatGPT", "Improve")],
    },
    "Claude answers → ChatGPT reviews → Claude revises ↺": {
        "opening": [("Claude", "Answer")],
        "loop": [("ChatGPT", "Review"), ("Claude", "Improve")],
    },
    "Ping-pong: both review & improve in turn ↺": {
        "opening": [("ChatGPT", "Answer")],
        "loop": [
            ("Claude", "Review"), ("Claude", "Improve"),
            ("ChatGPT", "Review"), ("ChatGPT", "Improve"),
        ],
    },
}

SYSTEM_BASE = (
    "You are {me}, collaborating with another AI ({other}) to produce the best "
    "possible answer to a user's question. Be accurate, specific and honest. "
    "Disagree when the other AI is wrong; agree when it is right."
)

ANSWER_PROMPT = """Question from the user:
<question>
{question}
</question>

Write a complete, high-quality answer."""

IMPROVE_PROMPT = """Question from the user:
<question>
{question}
</question>

Current answer (written by {author}):
<answer>
{answer}
</answer>
{feedback_block}
Produce an improved version of the answer. Fix every factual error, gap, or
unclear part{feedback_hint}. Keep what is already good. If you disagree with
a piece of feedback, keep the original and briefly note why at the very end
under a "Notes" heading. Output the full revised answer, not a diff."""

FEEDBACK_BLOCK = """
Latest review (by {reviewer}, score {score}/10):
<review>
{review}
</review>
"""

REVIEW_PROMPT = """Question from the user:
<question>
{question}
</question>

Answer to evaluate (written by {author}):
<answer>
{answer}
</answer>

Critically review this answer for correctness, completeness, clarity and
usefulness. Be strict: only give 9-10 if you would ship it as-is.

Respond with ONLY a JSON object, no other text:
{{"score": <integer 0-10>, "verdict": "<one sentence>", "issues": ["<specific problem and how to fix it>", ...]}}
Use an empty issues list if there is nothing left to fix."""


@dataclass
class Step:
    agent: str
    action: str
    content: str
    score: float | None = None
    issues: list[str] = field(default_factory=list)
    verdict: str = ""
    round: int = 0


@dataclass
class Result:
    final_answer: str
    final_author: str
    steps: list[Step]
    stopped_reason: str


def parse_review(text: str) -> tuple[float | None, str, list[str]]:
    """Pull score/verdict/issues out of a reviewer's (hopefully JSON) reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = data.get("score")
            score = float(score) if score is not None else None
            issues = [str(i) for i in data.get("issues") or []]
            return score, str(data.get("verdict", "")), issues
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    m = re.search(r"score\"?\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.I) or re.search(
        r"(\d+(?:\.\d+)?)\s*/\s*10", text
    )
    return (float(m.group(1)) if m else None), "", []


def format_review(score, verdict, issues) -> str:
    lines = [verdict] if verdict else []
    lines += [f"- {i}" for i in issues]
    return "\n".join(lines) or "(no details)"


def run_discussion(
    question: str,
    agents: dict,  # {"ChatGPT": provider, "Claude": provider}
    opening: list[tuple[str, str]],
    loop: list[tuple[str, str]],
    threshold: float,
    max_rounds: int,
    require_all_reviewers: bool,
    on_step_start: Callable[[str, str, int], None],
    on_stream: Callable[[Iterator[str]], str],
    on_step_done: Callable[[Step], None],
) -> Result:
    """Run the flow. UI hooks:
    on_step_start(agent, action, round) -> called before each step
    on_stream(chunks) -> renders the streaming text and returns the full text
    on_step_done(step) -> called with the finished step
    """
    steps: list[Step] = []
    answer, author = "", ""
    last_review: Step | None = None
    latest_scores: dict[str, float] = {}  # reviewer -> latest score
    reviewers_in_loop = {a for a, act in loop if act == "Review"}

    def other(name: str) -> str:
        return "Claude" if name == "ChatGPT" else "ChatGPT"

    def run_step(agent: str, action: str, rnd: int) -> Step:
        nonlocal answer, author, last_review
        system = SYSTEM_BASE.format(me=agent, other=other(agent))
        if action == "Answer" or (action == "Improve" and not answer):
            prompt = ANSWER_PROMPT.format(question=question)
            action = "Answer"
        elif action == "Improve":
            fb, hint = "", ""
            if last_review is not None:
                fb = FEEDBACK_BLOCK.format(
                    reviewer=last_review.agent,
                    score=_fmt(last_review.score),
                    review=format_review(
                        last_review.score, last_review.verdict, last_review.issues
                    ),
                )
                hint = ", especially the issues raised in the review"
            prompt = IMPROVE_PROMPT.format(
                question=question, author=author, answer=answer,
                feedback_block=fb, feedback_hint=hint,
            )
        else:  # Review
            if not answer:
                raise ValueError("A Review step needs an answer first. "
                                 "Start the flow with an Answer step.")
            prompt = REVIEW_PROMPT.format(question=question, author=author, answer=answer)

        on_step_start(agent, action, rnd)
        text = on_stream(agents[agent].stream(system, prompt))
        step = Step(agent=agent, action=action, content=text, round=rnd)

        if action == "Review":
            step.score, step.verdict, step.issues = parse_review(text)
            last_review = step
            if step.score is not None:
                latest_scores[agent] = step.score
        else:
            answer, author = text, agent
            last_review = None  # feedback has been consumed
        steps.append(step)
        on_step_done(step)
        return step

    def threshold_met() -> bool:
        if last_review is None or last_review.score is None:
            return False
        if require_all_reviewers and reviewers_in_loop:
            return all(
                latest_scores.get(r, -1) >= threshold for r in reviewers_in_loop
            )
        return last_review.score >= threshold

    for agent, action in opening:
        run_step(agent, action, 0)
        if action == "Review" and threshold_met():
            return Result(answer, author, steps, "Score threshold reached in opening")

    if not loop or not any(act == "Review" for _, act in loop):
        # Without a Review step there's no score to stop on: run the loop once.
        for agent, action in loop:
            run_step(agent, action, 1)
        return Result(answer, author, steps,
                      "Flow finished (loop has no Review step, so no scoring)")

    for rnd in range(1, max_rounds + 1):
        for agent, action in loop:
            run_step(agent, action, rnd)
            if action == "Review" and threshold_met():
                return Result(
                    answer, author, steps,
                    f"Score {_fmt(last_review.score)} ≥ {_fmt(threshold)} "
                    f"in round {rnd} — no remaining kinks",
                )
    return Result(answer, author, steps,
                  f"Stopped after max {max_rounds} rounds without reaching threshold")


def _fmt(x) -> str:
    if x is None:
        return "?"
    return f"{x:g}"
