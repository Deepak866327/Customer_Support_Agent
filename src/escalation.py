"""Rule-based AUTO_HANDLE / ESCALATE decision.

Deliberately not a learned risk model. With 200 golden examples there is nothing to
train a calibrated risk model on, and an opaque score cannot give the "stated reason"
the brief asks for. Every rule below is one readable condition, and the first one that
fires is the reason returned.

The rules are ordered by how expensive the mistake is, not by how often they fire:
safety conditions are checked before convenience ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.intents import OTHER, Taxonomy
from src.retrieval import Evidence, evidence_agreement

Action = Literal["AUTO_HANDLE", "ESCALATE"]

# Phrases where an automated reply is unacceptable regardless of how good the evidence
# looks, because a wrong answer creates legal, financial or safety exposure.
HIGH_RISK_RE = re.compile(
    r"(?:charge ?back|dispute|lawyer|legal action|sue\b|lawsuit|fraud|scam(?:med)?"
    r"|police|gdpr|data protection|complaint to|ombudsman|delete (?:my|all my) (?:account|data)"
    r"|identity theft|stolen card|my (?:credit|debit) card was)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EscalationConfig:
    """Thresholds. Defaults tuned on the golden set's dev half - see DECISIONS.md."""

    min_intent_confidence: float = 0.45
    min_top_similarity: float = 0.55
    min_evidence_agreement: float = 0.60
    min_evidence_count: int = 2
    max_handoff_share: float = 0.50


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    rule: str

    def as_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason, "rule": self.rule}


def decide(
    message: str,
    intent: str,
    intent_confidence: float,
    evidence: list[Evidence],
    taxonomy: Taxonomy,
    config: EscalationConfig | None = None,
) -> Decision:
    cfg = config or EscalationConfig()

    if HIGH_RISK_RE.search(message):
        return Decision(
            "ESCALATE",
            "Message contains a legal, fraud or payment-dispute signal that must be "
            "handled by a person.",
            "high_risk_language",
        )

    if intent == OTHER:
        return Decision(
            "ESCALATE",
            f"No intent reached the confidence threshold "
            f"({intent_confidence:.2f} < {cfg.min_intent_confidence:.2f}), so the "
            "request is not understood well enough to answer automatically.",
            "intent_abstained",
        )

    spec = taxonomy.get(intent)
    if spec.escalation == "escalate":
        return Decision(
            "ESCALATE",
            f"Intent '{intent}' always requires a human: {spec.escalation_reason}",
            "intent_requires_human",
        )

    if intent_confidence < cfg.min_intent_confidence:
        return Decision(
            "ESCALATE",
            f"Intent confidence {intent_confidence:.2f} is below the "
            f"{cfg.min_intent_confidence:.2f} threshold.",
            "low_confidence",
        )

    if len(evidence) < cfg.min_evidence_count:
        return Decision(
            "ESCALATE",
            f"Only {len(evidence)} similar historical resolution(s) found; at least "
            f"{cfg.min_evidence_count} are needed to answer from precedent.",
            "insufficient_evidence",
        )

    top_sim = max(e.similarity for e in evidence)
    if top_sim < cfg.min_top_similarity:
        return Decision(
            "ESCALATE",
            f"Closest historical case is only {top_sim:.2f} similar (threshold "
            f"{cfg.min_top_similarity:.2f}) - no sufficiently similar precedent exists.",
            "weak_retrieval",
        )

    agreement = evidence_agreement(evidence)
    if agreement < cfg.min_evidence_agreement:
        return Decision(
            "ESCALATE",
            f"Retrieved precedents disagree (only {agreement:.0%} share the same "
            "intent), so there is no consistent historical answer to follow.",
            "conflicting_evidence",
        )

    handoff_share = sum(e.was_handoff for e in evidence) / len(evidence)
    if handoff_share > cfg.max_handoff_share:
        return Decision(
            "ESCALATE",
            f"{handoff_share:.0%} of similar past cases were handed to a human agent, "
            "so this issue historically needed one.",
            "historically_human_handled",
        )

    return Decision(
        "AUTO_HANDLE",
        f"Intent '{intent}' predicted with {intent_confidence:.2f} confidence, "
        f"{len(evidence)} consistent precedents ({agreement:.0%} agreement, closest "
        f"{top_sim:.2f}), and only {handoff_share:.0%} of them needed a human.",
        "confident_and_grounded",
    )
