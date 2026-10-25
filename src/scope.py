"""Keep the chat assistant on Spotify customer support and nothing else.

Out-of-scope requests (coding, homework, essays, general knowledge, attempts to change
the assistant's instructions) get a fixed reply, not a generated one, so there is
nothing for a clever prompt to steer. Three checks run before the LLM, cheapest first;
`src.chat` adds two more around it (the model can flag a request, and its reply is
inspected before it is sent).

Measured on the 200 golden messages vs 25 hand-written off-domain probes (top cosine
similarity of the message to the 13,347 indexed support cases):

    golden  min 0.41  p5 0.50  median 0.69
    off     min 0.25         median 0.36  max 0.59 ("my netflix account isn't working")

MIN_SUPPORT_SIMILARITY = 0.40 flags 0/200 golden messages and 0/8 typical follow-ups
("it's on android") and catches 64% of the probes. The pattern rules and the model
check cover the rest - similarity alone cannot, because "how do i reset my iphone" is
phrased exactly like a support question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_SUPPORT_SIMILARITY = 0.40
# Below this many words the similarity check is skipped: "ok", "yes", "thanks" carry
# no topic, and a greeting is handled by the clarifying path instead.
MIN_WORDS_FOR_SIMILARITY = 4

OUT_OF_SCOPE_REPLY = (
    "Sorry, I can only help with Spotify - things like your account, subscription and "
    "payments, the app on your devices, and music, podcasts or playlists. Is there "
    "anything Spotify-related I can help you with?"
)

# "promo code", "gift card code" etc. are support topics; removed before code checks.
_IN_DOMAIN_CODE = re.compile(
    r"\b(?:promo|promotional|gift(?: card)?|discount|voucher|redeem|redemption|"
    r"activation|verification|security|scan|qr|spotify|country|zip|postal|error)\s+codes?\b",
    re.IGNORECASE,
)

_RULES: list[tuple[str, str, re.Pattern]] = [
    (
        "prompt_injection",
        "The message tries to change the assistant's instructions or role.",
        re.compile(
            r"\b(?:ignore|disregard|forget)\b.{0,20}\b(?:previous|prior|above|earlier|"
            r"all|your)\b.{0,15}\b(?:instructions?|prompts?|rules)\b"
            r"|\byou are now\b|\bpretend (?:to be|you(?:'re| are))\b|\bact as (?:a|an|my)\b"
            r"|\bsystem prompt\b|\bjailbreak\b|\bdeveloper mode\b",
            re.IGNORECASE,
        ),
    ),
    (
        "programming_request",
        "The message asks for programming help, which is outside customer support.",
        re.compile(
            r"\b(?:python|javascript|typescript|java|c\+\+|c#|golang|kotlin|php|sql|"
            r"html|css|regex|node\.?js|reactjs|react component)\b"
            r"|\b(?:write|generate|create|give me|fix|debug|refactor|optimi[sz]e|explain)\b"
            r".{0,40}\b(?:code|script|program|function|algorithm|class|query|snippet)\b"
            r"|\b(?:null pointer|stack ?trace|syntax error|compile error|linked list|"
            r"leetcode|binary tree)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "writing_or_homework",
        "The message asks for writing, homework or general tasks outside customer support.",
        re.compile(
            r"\b(?:write|compose|create|generate)\b.{0,30}\b(?:poem|essay|story|"
            r"cover letter|resume|article|blog post|speech|joke)\b"
            r"|\b(?:homework|assignment|equation|integral|derivative)\b"
            r"|\btell me a joke\b|\bsummari[sz]e the (?:plot|book|novel|chapter)\b",
            re.IGNORECASE,
        ),
    ),
]

# A reply that looks like code is never sent, whatever the reason it was produced.
_CODE_IN_REPLY = re.compile(
    r"```|^\s*(?:def|class|import|function|const|let|var|public|#include)\b"
    r"|^\s*SELECT\b.+\bFROM\b|<\s*(?:script|html|div)\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class ScopeVerdict:
    rule: str
    reason: str


def check_message(message: str, top_similarity: float | None) -> ScopeVerdict | None:
    """A verdict if `message` is out of scope, else None.

    `top_similarity` is the message's closest match among indexed support cases; pass
    None to skip that check.
    """
    text = _IN_DOMAIN_CODE.sub(" ", message)
    for rule, reason, pattern in _RULES:
        if pattern.search(text):
            return ScopeVerdict(rule, reason)
    if (
        top_similarity is not None
        and len(message.split()) >= MIN_WORDS_FOR_SIMILARITY
        and top_similarity < MIN_SUPPORT_SIMILARITY
    ):
        return ScopeVerdict(
            "unrelated_to_support",
            f"The message is unlike any past support case (closest {top_similarity:.2f}, "
            f"threshold {MIN_SUPPORT_SIMILARITY:.2f}).",
        )
    return None


def reply_looks_like_code(reply: str) -> bool:
    return bool(_CODE_IN_REPLY.search(reply))
