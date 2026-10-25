"""Intent taxonomy and the rule-based weak labeller.

The taxonomy lives in `intents.yaml` so it can be read and argued with by a human.
This module loads it and applies the patterns to produce *weak* training labels.

Why weak labels at all: the golden set is hand-labelled and only 200 examples, far too
few to train on. Rules give ~13k labelled training messages for free and deterministically.
Their cost is a real ceiling on the classifier, measured honestly in the report:
`evaluation/run.py` reports weak-label precision against the human golden labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

TAXONOMY_PATH = Path(__file__).with_name("intents.yaml")
OTHER = "other"


@dataclass(frozen=True)
class Intent:
    name: str
    description: str
    escalation: str  # "auto" | "escalate" | "conditional"
    escalation_reason: str
    examples: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class Taxonomy:
    intents: tuple[Intent, ...]

    @property
    def names(self) -> list[str]:
        return [i.name for i in self.intents]

    def get(self, name: str) -> Intent:
        for i in self.intents:
            if i.name == name:
                return i
        raise KeyError(name)

    @property
    def trainable_names(self) -> list[str]:
        """The 11 real intents. `other` is excluded - see `seed_label`."""
        return [i.name for i in self.intents if i.name != OTHER]

    def seed_label(self, text: str) -> str | None:
        """High-precision weak label, or None when no rule fires.

        None means *unlabelled*, NOT `other`. Conflating the two was a design error:
        57.5% of opening messages match no rule, and sampling them shows they mostly
        DO have a clear intent that keywords simply cannot see ("I thought Liam's new
        song was coming out today, can't see it on Spotify"). Training a literal
        `other` class on them would teach the classifier that the majority of real
        traffic is intentless, and would hand the trivial baseline a fake 57.5%.

        So unmatched messages are dropped from training, and `other` exists only as an
        *abstention* the classifier emits at inference when no intent clears the
        confidence threshold (see src/classifier.py).
        """
        if not text:
            return None
        for intent in self.intents:
            for pat in intent.patterns:
                if pat.search(text):
                    return intent.name
        return None

    def seed_label_many(self, texts) -> list[str | None]:
        return [self.seed_label(t) for t in texts]


@lru_cache(maxsize=1)
def load_taxonomy(path: str | Path = TAXONOMY_PATH) -> Taxonomy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    intents = []
    for name, spec in raw["intents"].items():
        intents.append(
            Intent(
                name=name,
                description=" ".join(spec["description"].split()),
                escalation=spec["escalation"],
                escalation_reason=" ".join(spec["escalation_reason"].split()),
                examples=tuple(spec.get("examples") or ()),
                patterns=tuple(
                    re.compile(p, re.IGNORECASE) for p in (spec.get("patterns") or ())
                ),
            )
        )
    return Taxonomy(tuple(intents))
