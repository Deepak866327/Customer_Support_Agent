"""The support agent: classify -> retrieve -> draft -> decide.

    from src.agent import SupportAgent
    agent = SupportAgent.load()
    print(agent.handle("my app keeps crashing on android"))
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

from src.classifier import EmbeddingClassifier
from src.escalation import Decision, EscalationConfig, decide
from src.generator import Draft, generate
from src.intents import Taxonomy, load_taxonomy
from src.llm import LLMClient
from src.retrieval import Evidence, ResolutionIndex

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
DEFAULT_K = 5


@dataclass
class AgentResponse:
    customer_message: str
    intent: str
    intent_confidence: float
    evidence: list[Evidence]
    decision: Decision
    draft: Draft | None = None

    def as_dict(self) -> dict:
        return {
            "customer_message": self.customer_message,
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 4),
            "action": self.decision.action,
            "reason": self.decision.reason,
            "rule": self.decision.rule,
            "reply": self.draft.reply if self.draft else None,
            "reply_confidence": self.draft.confidence if self.draft else None,
            "evidence_ids": self.draft.evidence_ids if self.draft else [],
            "evidence": [e.as_dict() for e in self.evidence],
        }

    def __str__(self) -> str:
        head = f"[{self.decision.action}] {self.intent} ({self.intent_confidence:.2f})"
        body = self.draft.reply if self.draft else "(no draft - escalated)"
        return f"{head}\n  reason: {self.decision.reason}\n  reply : {body}"


@dataclass
class SupportAgent:
    classifier: EmbeddingClassifier
    index: ResolutionIndex
    taxonomy: Taxonomy = field(default_factory=load_taxonomy)
    config: EscalationConfig = field(default_factory=EscalationConfig)
    llm: LLMClient = field(default_factory=LLMClient)
    k: int = DEFAULT_K

    @classmethod
    def load(cls, cache: Path = CACHE, **kwargs) -> "SupportAgent":
        with (cache / "classifier.pkl").open("rb") as fh:
            clf = pickle.load(fh)
        return cls(classifier=clf, index=ResolutionIndex.load(), **kwargs)

    def handle(self, message: str, draft_reply: bool = True) -> AgentResponse:
        """Run the full pipeline on one customer message.

        A reply is drafted even when the decision is ESCALATE: the draft becomes the
        suggested response a human agent edits, which is how these systems are actually
        deployed, and it lets reply quality be evaluated on the whole golden set rather
        than only on the auto-handled slice.
        """
        (intent,), (confidence,) = self.classifier.predict([message], tag="live")
        evidence = self.index.search(message, k=self.k, intent=intent)
        decision = decide(
            message, intent, float(confidence), evidence, self.taxonomy, self.config
        )
        draft = (
            generate(message, intent, evidence, self.llm) if draft_reply else None
        )
        return AgentResponse(
            customer_message=message,
            intent=intent,
            intent_confidence=float(confidence),
            evidence=evidence,
            decision=decision,
            draft=draft,
        )
