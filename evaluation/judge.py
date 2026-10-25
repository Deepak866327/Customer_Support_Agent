"""LLM-as-judge for reply quality, on a 1-5 rubric.

The judge sees the customer message, the retrieved precedents, the reply under test,
and the reply Spotify actually sent. It does NOT see which system produced the reply,
so the same prompt scores the proposed system and its baselines identically.

Judge scores are treated as an unvalidated instrument until proven otherwise:
`scripts/judge_validation.py` compares them against human scores on 40 examples and
reports quadratic-weighted Cohen's kappa. Read that before believing these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.llm import JUDGE_MODEL, LLMClient, complete_json
from src.retrieval import Evidence

CRITERIA = ("correctness", "groundedness", "consistency", "helpfulness", "no_unsupported_claims")

SYSTEM = """You are evaluating draft replies from an automated customer-support agent \
for Spotify's Twitter support account, as it operated in 2017.

Score the CANDIDATE REPLY on five criteria, each an integer 1-5.

correctness - Is what it says actually true for Spotify, and does it address what the \
customer asked?
  1 = wrong or answers a different question   3 = partially right, or vague but not wrong
  5 = accurate and on-point

groundedness - Is every claim supported by the PRECEDENTS shown?
  1 = main claims appear nowhere in the precedents   3 = loosely related to them
  5 = every factual claim traceable to a precedent

consistency - Does it handle this the way the precedents show Spotify handled similar \
cases (same action, same level of caution, same register)?
  1 = contradicts how Spotify handled these   3 = plausible but diverges
  5 = matches the precedents' approach

helpfulness - Does it move the customer forward?
  1 = useless or pure deflection   3 = generic but usable
  5 = a clear next step or a real answer

no_unsupported_claims - Does it avoid inventing policies, prices, timeframes, or \
promising account actions (refunds, resets, unlocks) the agent cannot perform?
  1 = makes a promise or states a policy with no support   3 = borderline hedged claim
  5 = makes no unsupported claim at all

Judge only the reply in front of you. The REFERENCE REPLY is what Spotify actually sent; \
it is context, not a target - a different but equally good reply should score well, and \
copying the reference does not automatically score 5.

Return ONLY JSON:
{"correctness": n, "groundedness": n, "consistency": n, "helpfulness": n, \
"no_unsupported_claims": n, "rationale": "<one sentence>"}"""


@dataclass
class JudgeScore:
    scores: dict[str, int]
    rationale: str = ""

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / len(self.scores)

    def as_dict(self) -> dict:
        return {**self.scores, "mean": round(self.mean, 3), "rationale": self.rationale}


@dataclass
class Judge:
    client: LLMClient = field(default_factory=lambda: LLMClient(model=JUDGE_MODEL))

    def score(
        self,
        customer_message: str,
        candidate_reply: str,
        evidence: list[Evidence],
        reference_reply: str,
    ) -> JudgeScore:
        prompt = _build_prompt(customer_message, candidate_reply, evidence, reference_reply)
        data = complete_json(self.client, prompt, system=SYSTEM, max_tokens=400)
        scores = {}
        for c in CRITERIA:
            try:
                scores[c] = max(1, min(5, int(round(float(data.get(c, 3))))))
            except (TypeError, ValueError):
                scores[c] = 3
        return JudgeScore(scores=scores, rationale=str(data.get("rationale", ""))[:300])


def _build_prompt(
    customer_message: str,
    candidate_reply: str,
    evidence: list[Evidence],
    reference_reply: str,
) -> str:
    lines = [f"CUSTOMER MESSAGE:\n{customer_message}", "\nPRECEDENTS:"]
    if not evidence:
        lines.append("(none were retrieved for this reply)")
    for e in evidence:
        lines.append(
            f"\n[{e.evidence_id}] past customer: {e.customer_message}"
            f"\n     spotify replied: {e.resolution}"
        )
    lines += [
        f"\nCANDIDATE REPLY (score this):\n{candidate_reply or '(empty)'}",
        f"\nREFERENCE REPLY actually sent by Spotify (context only):\n{reference_reply}",
        "\nReturn the JSON scores.",
    ]
    return "\n".join(lines)
