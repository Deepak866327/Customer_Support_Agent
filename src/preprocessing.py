"""Load the raw twcs dump, reconstruct conversations, and build the SpotifyCares corpus.

The raw file is a reply *forest*: every row is a tweet, and `in_response_to_tweet_id`
points at its parent. We recover conversations as connected components of that graph.
See DATA_ANALYSIS.md for the measured properties this module relies on.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BRAND = "SpotifyCares"

# Brand replies that only push the customer to a private channel. Measured at 30.8% of
# SpotifyCares replies; this is the observed "a human took over" signal, not noise.
HANDOFF_RE = re.compile(
    r"(?:\bdm\b|\bdms\b|direct message|private message|\bpm us\b|message us"
    r"|drop us a (?:dm|message|note)|send us a (?:dm|pm|message)|shoot us a"
    r"|follow(?:ing)? (?:and|&) dm)",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
# Spotify agents sign off with "/JP" or "^AB"; keeping these leaks agent identity into
# the text and adds nothing, so they are stripped.
_SIGNATURE_RE = re.compile(r"\s*[/^]\s?[A-Z]{1,3}\s*$")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a tweet for embedding/classification.

    URLs become a `<link>` sentinel rather than vanishing, because "here is a link"
    is itself meaningful support behaviour we do not want to erase.
    """
    if not isinstance(text, str):
        return ""
    out = html.unescape(text)
    out = _URL_RE.sub(" <link> ", out)
    out = _MENTION_RE.sub(" ", out)
    out = _SIGNATURE_RE.sub("", out)
    out = out.replace("’", "'").replace("“", '"').replace("”", '"')
    return _WS_RE.sub(" ", out).strip()


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read twcs.csv with explicit dtypes and a parsed timestamp."""
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "string",
            "text": "string",
            "response_tweet_id": "string",
        },
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )
    df["inbound"] = df["inbound"].str.lower().eq("true")
    df["text"] = df["text"].fillna("")
    df["created_at"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    return df


def assign_conversations(df: pd.DataFrame) -> pd.DataFrame:
    """Attach `conv_id` = connected component over the reply graph (union-find).

    Preferred over walking `response_tweet_id` forward because that column is a
    comma-separated multi-value field and threads branch.
    """
    position = pd.Series(np.arange(len(df)), index=df["tweet_id"].values)
    parent = np.arange(len(df), dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    has_parent = df["in_response_to_tweet_id"].notna()
    child_pos = np.arange(len(df))[has_parent.values]
    parent_ids = df.loc[has_parent, "in_response_to_tweet_id"].astype("int64").values
    parent_pos = position.reindex(parent_ids).values
    resolvable = ~pd.isna(parent_pos)

    for child, par in zip(child_pos[resolvable], parent_pos[resolvable].astype(np.int64)):
        ra, rb = find(int(child)), find(int(par))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    out = df.copy()
    out["conv_id"] = [find(i) for i in range(len(df))]
    return out


@dataclass(frozen=True)
class CorpusStats:
    conversations: int
    customer_messages: int
    brand_messages: int
    pairs: int
    handoff_rate: float


def build_brand_corpus(df: pd.DataFrame, brand: str = BRAND) -> pd.DataFrame:
    """All tweets belonging to conversations the brand participated in."""
    brand_convs = df.loc[df["author_id"] == brand, "conv_id"].unique()
    sub = df[df["conv_id"].isin(brand_convs)].copy()
    sub = sub.sort_values(["conv_id", "created_at"], kind="stable")
    sub["clean_text"] = sub["text"].map(clean_text)
    return sub


def build_pairs(corpus: pd.DataFrame, brand: str = BRAND) -> pd.DataFrame:
    """One row per (customer message -> brand reply) adjacency.

    `is_opening` marks the customer's first message in the conversation: that is the
    message an agent would actually triage, and mid-thread fragments ("iPhone 6
    ios11.1.1") are meaningless in isolation.
    """
    lookup = corpus[["tweet_id", "author_id", "inbound", "text", "clean_text"]].rename(
        columns={
            "tweet_id": "parent_id",
            "author_id": "parent_author",
            "inbound": "parent_inbound",
            "text": "customer_text_raw",
            "clean_text": "customer_text",
        }
    )
    replies = corpus[corpus["in_response_to_tweet_id"].notna()]
    merged = replies.merge(
        lookup, left_on="in_response_to_tweet_id", right_on="parent_id", how="inner"
    )
    pairs = merged[(merged["author_id"] == brand) & (merged["parent_inbound"])].copy()

    first_customer = (
        corpus[corpus["inbound"]].groupby("conv_id", sort=False)["tweet_id"].first()
    )
    opening_ids = set(first_customer.values.tolist())

    handoff_by_conv = (
        corpus[corpus["author_id"] == brand]
        .assign(h=lambda d: d["text"].str.contains(HANDOFF_RE, regex=True))
        .groupby("conv_id")["h"]
        .any()
    )

    pairs = pairs.rename(
        columns={
            "tweet_id": "reply_id",
            "text": "reply_text_raw",
            "clean_text": "reply_text",
            "created_at": "reply_at",
        }
    )
    pairs["pair_id"] = pairs["reply_id"].astype(str)
    pairs["is_opening"] = pairs["parent_id"].isin(opening_ids)
    pairs["reply_is_handoff"] = pairs["reply_text_raw"].str.contains(HANDOFF_RE, regex=True)
    pairs["conv_handoff"] = pairs["conv_id"].map(handoff_by_conv).fillna(False)
    pairs["reply_words"] = pairs["reply_text"].str.split().str.len()
    # "Substantive" = a reply that could actually be reused as evidence for a future
    # customer: not a bare handoff, and long enough to carry content.
    pairs["is_substantive"] = (~pairs["reply_is_handoff"]) & (pairs["reply_words"] >= 12)

    cols = [
        "pair_id",
        "conv_id",
        "parent_id",
        "reply_id",
        "customer_text",
        "customer_text_raw",
        "reply_text",
        "reply_text_raw",
        "reply_at",
        "is_opening",
        "reply_is_handoff",
        "conv_handoff",
        "reply_words",
        "is_substantive",
    ]
    return pairs[cols].sort_values("reply_at", kind="stable").reset_index(drop=True)


def time_split(
    pairs: pd.DataFrame, cutoff: str = "2017-11-25"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by conversation using a time cutoff.

    Conversation-level, not tweet-level: two tweets from one thread share near-identical
    wording, so a random split would let the classifier and the retriever see the eval
    thread during training. The cutoff is on the conversation's *first* timestamp so a
    thread never straddles the boundary.
    """
    conv_start = pairs.groupby("conv_id")["reply_at"].min()
    boundary = pd.Timestamp(cutoff, tz="UTC")
    train_convs = set(conv_start[conv_start < boundary].index)
    train = pairs[pairs["conv_id"].isin(train_convs)].copy()
    evalp = pairs[~pairs["conv_id"].isin(train_convs)].copy()
    return train, evalp


def corpus_stats(corpus: pd.DataFrame, pairs: pd.DataFrame, brand: str = BRAND) -> CorpusStats:
    return CorpusStats(
        conversations=int(corpus["conv_id"].nunique()),
        customer_messages=int(corpus["inbound"].sum()),
        brand_messages=int((corpus["author_id"] == brand).sum()),
        pairs=len(pairs),
        handoff_rate=float(pairs["conv_handoff"].mean()),
    )
