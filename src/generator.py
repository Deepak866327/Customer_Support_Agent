"""Draft a reply grounded in retrieved historical resolutions."""

from __future__ import annotations

from dataclasses import dataclass

from src.llm import LLMClient, complete_json
from src.retrieval import Evidence

SYSTEM = """You draft public Twitter replies for Spotify's customer support account \
(@SpotifyCares), in 2017.

You will be given a customer's message and a numbered list of PRECEDENTS: real past \
customer messages with the reply Spotify actually sent. Your reply must be consistent \
with those precedents.

Hard rules:
- Ground every factual claim in the precedents. If the precedents do not establish \
something, do not say it.
- Never invent a policy, price, timeframe, refund, or entitlement.
- Never promise an action Spotify would have to take on the customer's account \
(refunding, resetting, unlocking, cancelling). You cannot see their account.
- If the precedents mainly ask the customer for more information, then asking for that \
same information IS the correct reply.
- Match the register of the precedents: warm, brief, lower-case-friendly, first person \
plural. Under 280 characters.
- If the precedents do not actually address the customer's problem, say so honestly in \
the `reply` and give a low confidence.

Return ONLY a JSON object:
{"reply": "<the reply text>", "confidence": <0.0-1.0>, "evidence_ids": ["<ids you used>"]}

`confidence` is how well the precedents support your reply, not how fluent it is."""


@dataclass(frozen=True)
class Draft:
    reply: str
    confidence: float
    evidence_ids: list[str]

    def as_dict(self) -> dict:
        return {
            "reply": self.reply,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
        }


def build_prompt(message: str, intent: str, evidence: list[Evidence]) -> str:
    lines = [f"CUSTOMER MESSAGE:\n{message}", f"\nPREDICTED INTENT: {intent}", "\nPRECEDENTS:"]
    if not evidence:
        lines.append("(none found)")
    for e in evidence:
        lines.append(
            f"\n[{e.evidence_id}] (similarity {e.similarity:.2f}, intent {e.intent})"
            f"\n  past customer: {e.customer_message}"
            f"\n  spotify replied: {e.resolution}"
        )
    lines.append("\nDraft the reply as JSON.")
    return "\n".join(lines)


def generate(
    message: str,
    intent: str,
    evidence: list[Evidence],
    client: LLMClient | None = None,
) -> Draft:
    client = client or LLMClient()
    data = complete_json(
        client, build_prompt(message, intent, evidence), system=SYSTEM, max_tokens=500
    )
    return draft_from_json(data, evidence)


def draft_from_json(data: dict, evidence: list[Evidence]) -> Draft:
    """Validate a model's JSON into a Draft; cited ids not in `evidence` are dropped."""
    valid = {e.evidence_id for e in evidence}
    cited = [str(i) for i in data.get("evidence_ids", []) if str(i) in valid]
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Draft(
        reply=str(data.get("reply", "")).strip(),
        confidence=max(0.0, min(1.0, conf)),
        evidence_ids=cited,
    )
