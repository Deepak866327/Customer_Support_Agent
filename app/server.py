"""Customer-facing chat server for the support agent.

    python -m app.server            # http://127.0.0.1:8000
    uvicorn app.server:app --reload

The server is stateless: the browser sends the whole conversation on every turn, so a
restart loses nothing and there is no session store to secure. Generation needs the
provider key in .env; without one, every turn degrades to a clarifying question or a
handoff, never to an ungrounded answer.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat import ChatAgent, Turn  # noqa: E402
from src.embeddings import encode  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_TURNS = 40
MAX_CHARS = 2000

log = logging.getLogger("app")
agent: ChatAgent | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent
    agent = ChatAgent.load()
    encode(["warm-up"], use_cache=False)  # load the encoder now, not on the first message
    log.info("chat agent ready (%d indexed resolutions)", len(agent.index))
    yield


app = FastAPI(title="Support chat", lifespan=lifespan)


class TurnIn(BaseModel):
    role: Literal["customer", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_CHARS)
    action: Literal["AUTO_HANDLE", "CLARIFY", "ESCALATE", "OUT_OF_SCOPE"] | None = None


class ChatRequest(BaseModel):
    messages: list[TurnIn] = Field(min_length=1, max_length=MAX_TURNS)


@app.get("/api/health")
def health() -> dict:
    return {"ok": agent is not None, "model": agent.llm.model if agent else None}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    # Sync on purpose: the pipeline is CPU-bound, so FastAPI runs it in its threadpool.
    if agent is None:
        raise HTTPException(503, "agent is still loading")
    if req.messages[-1].role != "customer":
        raise HTTPException(422, "the last message must be from the customer")
    turns = [Turn(t.role, t.content, t.action) for t in req.messages]
    return agent.respond(turns).as_dict()


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
