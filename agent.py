"""사람이 승인하는 인적공제 상담 에이전트 (LangGraph + interrupt).

질문 → classify(카테고리 판정) ─(OTHER · 확신도 0.5 미만)→ handoff(상담원에게 넘김)
     → retrieve(카테고리별 근거 조회) → answer(근거만으로 답변) ─(NO_ANSWER)→ handoff
     → verify(숫자·근거 번호 검사, 위반이면 answer 1번 더 → 또 위반이면 handoff)
     → check(승인 기준 계산)
          ├ 기준에 안 걸림 → send(고객에게 자동 발송)
          └ 기준에 걸림   → review(멈춤: 담당자 답을 기다림)
                              ├ approve  → send(그대로 발송)
                              ├ edit     → send(고친 답 발송)
                              ├ reject   → rejected(발송 안 함, 상담원이 직접 처리)
                              └ rejudge  → answer(담당자 지시를 붙여 다시 답변) → verify → check → review

멈춘 건은 SqliteSaver(store/checkpoints.sqlite)에 저장되어, 프로그램을 껐다 켜도 이어서 처리할 수 있다.
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

import context as ctx
import criteria
from prompts import ANSWER_SYSTEM, RETRY_NOTE, ROUTER_SYSTEM

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
STORE = Path(os.environ.get("HITL_STORE", HERE / "store"))   # 승인 대기 건을 저장하는 폴더

ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gpt-5.6-luna")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "gpt-5.6-terra")

HANDOFF_CONF = 0.5     # 확신도가 이보다 낮으면 답하지 않고 상담원에게 넘긴다 (라우팅 과제와 같은 값)
MAX_ATTEMPTS = 2       # 검증 위반 시 답변을 쓰는 최대 횟수
MAX_REJUDGE = 2        # 담당자가 "다시 판정"을 요청할 수 있는 최대 횟수
EXPIRE_DAYS = 3        # 승인 대기가 이 기간을 넘으면 자동 반려 → 상담원이 직접 처리

Category = Literal["FAMILY", "INCOME", "EXTRA", "CREDIT", "OTHER"]


class RouteDecision(BaseModel):
    categories: list[Category] = Field(description="필요한 카테고리 1~2개. 중심이 되는 것을 맨 앞에")
    confidence: float = Field(ge=0, le=1, description="판정이 맞을 확률")
    reason: str = Field(description="한 문장 이유")


_router = None
_answerer = None


def router():
    global _router
    if _router is None:
        _router = init_chat_model(ROUTER_MODEL, model_provider="openai", temperature=0,
                                  timeout=60, max_retries=2).with_structured_output(RouteDecision)
    return _router


def answerer():
    global _answerer
    if _answerer is None:
        _answerer = init_chat_model(ANSWER_MODEL, model_provider="openai", temperature=0,
                                    timeout=90, max_retries=2)
    return _answerer


TOOL_OF = {"FAMILY": "lookup_family", "INCOME": "lookup_income",
           "EXTRA": "lookup_extra", "CREDIT": "lookup_credit"}
HANDOFF_MESSAGE = "이 질문은 인적공제 자료로 답할 수 없어 상담원에게 넘기겠습니다."


class State(TypedDict, total=False):
    question: str
    received_at: str
    categories: list[str]
    confidence: float
    reason: str
    tools_called: list[str]
    context: str
    chunk_ids: list[str]
    answer: str
    attempts: int                # 이번 답변을 몇 번째에 썼는지 (검증 위반이면 2)
    problems: list[str]
    checks: list[dict]
    stop_reasons: list[str]      # 걸린 승인 기준 (criteria.CANDIDATES의 키)
    rejudges: int                # 다시 판정한 횟수
    instruction: str             # 다시 판정 때 담당자 지시
    decisions: list[dict]        # 담당자 답 기록
    status: str                  # sent | rejected | handoff
    sent_answer: str             # 고객에게 실제로 나간 글
    sent_by: str                 # 자동 | 승인 | 수정 후 승인
    finished_at: str
    handoff_reason: str
    reject_reason: str


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------- ① 판정 ----------
def classify(state: State) -> State:
    d: RouteDecision = router().invoke([("system", ROUTER_SYSTEM), ("human", state["question"])])
    cats = list(dict.fromkeys(d.categories))[:2] or ["OTHER"]
    if "OTHER" in cats:
        cats = ["OTHER"]
    return {"categories": cats, "confidence": d.confidence, "reason": d.reason,
            "tools_called": [], "attempts": 0, "checks": [], "rejudges": 0, "decisions": []}


def gate_next(state: State) -> str:
    if state["categories"] == ["OTHER"] or state["confidence"] < HANDOFF_CONF:
        return "handoff"
    return "retrieve"


# ---------- ② 근거 조회 ----------
def retrieve(state: State) -> State:
    text, ids = ctx.build_context(state["categories"])
    called = list(state.get("tools_called", [])) + [TOOL_OF[c] for c in state["categories"]]
    return {"tools_called": called, "context": text, "chunk_ids": ids}


# ---------- ③ 답변 ----------
def answer(state: State) -> State:
    msgs = [("system", ANSWER_SYSTEM),
            ("human", f"[근거]\n{state['context']}\n\n[질문]\n{state['question']}")]
    if state.get("instruction") or state.get("problems"):
        msgs.append(("ai", state["answer"]))
        notes = []
        if state.get("instruction"):        # 담당자가 다시 판정을 요청한 경우
            notes.append(f"[담당자 지시]\n{state['instruction']}\n위 지시에 따라 [근거]만으로 다시 답하십시오.")
        if state.get("problems"):           # 검증 위반으로 다시 쓰는 경우
            notes.append(RETRY_NOTE.format(problems="\n".join(f"- {p}" for p in state["problems"])))
        msgs.append(("human", "\n\n".join(notes)))
    out = answerer().invoke(msgs)
    text = out.content if isinstance(out.content, str) else "".join(
        b.get("text", "") for b in out.content if isinstance(b, dict))
    return {"answer": text.strip(), "attempts": state.get("attempts", 0) + 1}


def answer_next(state: State) -> str:
    return "handoff" if "NO_ANSWER" in state["answer"] else "verify"


# ---------- ④ 검증 (코드) ----------
CITE = re.compile(r"\[?((?:QA|KW)-\d{2,3})\]?")


def numbers_in(text: str) -> set[int]:
    text = CITE.sub(" ", text)                          # 근거 번호 안의 숫자는 빼고 본다
    text = re.sub(r"(?<=\d),(?=\d)", "", text)          # 5,166,667 → 5166667
    return {int(m) for m in re.findall(r"\d+", text)}


def guardrail(answer_text: str, context_text: str, question: str, chunk_ids: list[str],
              min_check: int = 10) -> dict:
    allowed = numbers_in(context_text) | numbers_in(question)
    found = numbers_in(answer_text)
    bad_numbers = sorted(n for n in found if n >= min_check and n not in allowed)
    cited = list(dict.fromkeys(CITE.findall(answer_text)))
    bad_cites = [c for c in cited if c not in chunk_ids]
    problems = []
    if bad_numbers:
        problems.append(f"근거에 없는 숫자: {bad_numbers}")
    if bad_cites:
        problems.append(f"조립한 근거에 없는 번호: {bad_cites}")
    if not cited:
        problems.append("근거 번호가 없음")
    return {"ok": not problems, "problems": problems, "numbers": sorted(found),
            "cited": cited, "bad_numbers": bad_numbers, "bad_cites": bad_cites}


def verify(state: State) -> State:
    g = guardrail(state["answer"], state["context"], state["question"], state["chunk_ids"])
    checks = list(state.get("checks", [])) + [{"attempt": state["attempts"], **g}]
    return {"checks": checks, "problems": [] if g["ok"] else g["problems"]}


def verify_next(state: State) -> str:
    if not state["problems"]:
        return "check"
    return "answer" if state["attempts"] < MAX_ATTEMPTS else "handoff"


# ---------- ⑤ 승인 기준 계산 ----------
def check(state: State) -> State:
    return {"stop_reasons": criteria.stop_reasons(state)}


def check_next(state: State) -> str:
    return "review" if state["stop_reasons"] else "send"


# ---------- ⑥ 사람 승인 (멈춤) ----------
def review(state: State) -> State:
    """담당자에게 묻고 답이 올 때까지 멈춘다.

    재개하면 이 단계가 첫 줄부터 다시 실행되므로, interrupt 앞에는 바깥으로 나가는 동작을 두지 않는다.
    """
    d = interrupt({
        "질문": state["question"],
        "AI 답변": state["answer"],
        "AI 판정": f"{', '.join(state['categories'])} (확신도 {state['confidence']:.2f}) — {state['reason']}",
        "근거 번호": state["checks"][-1]["cited"],
        "멈춘 이유": [criteria.describe(k) for k in state["stop_reasons"]],
        "통과시키면": "이 답변이 고객에게 그대로 발송됩니다.",
    })
    d = {**d, "at": now()}
    upd: State = {}
    if d["action"] == "rejudge":
        d["accepted"] = state.get("rejudges", 0) < MAX_REJUDGE   # 횟수를 넘으면 반려로 처리
        if d["accepted"]:
            upd.update({"rejudges": state.get("rejudges", 0) + 1, "instruction": d["instruction"],
                        "attempts": 0, "problems": []})
    upd["decisions"] = list(state.get("decisions", [])) + [d]
    return upd


def review_next(state: State) -> str:
    d = state["decisions"][-1]
    if d["action"] in ("approve", "edit"):
        return "send"
    if d["action"] == "rejudge" and d["accepted"]:
        return "answer"
    return "rejected"


# ---------- 끝 단계 ----------
def send(state: State) -> State:
    """고객에게 답을 보낸다. (데모라 실제 발송 대신 상태에 남긴다)"""
    last = state.get("decisions") or []
    if not state.get("stop_reasons"):
        by, text = "자동", state["answer"]
    elif last[-1]["action"] == "edit":
        by, text = "수정 후 승인", last[-1]["text"]
    else:
        by, text = "승인", state["answer"]
    return {"status": "sent", "sent_by": by, "sent_answer": text, "finished_at": now()}


def rejected(state: State) -> State:
    d = state["decisions"][-1]
    why = d.get("reason") or ("다시 판정 횟수 초과" if d["action"] == "rejudge" else "")
    return {"status": "rejected", "reject_reason": why, "finished_at": now()}


def handoff(state: State) -> State:
    if state.get("categories") == ["OTHER"]:
        why = "범위 밖(OTHER)"
    elif state.get("confidence", 1) < HANDOFF_CONF and not state.get("answer"):
        why = f"확신도 {state.get('confidence'):.2f} < {HANDOFF_CONF}"
    elif "NO_ANSWER" in state.get("answer", ""):
        why = "근거에 답이 없음"
    else:
        why = "검증 위반이 고쳐지지 않음"
    return {"tools_called": list(state.get("tools_called", [])) + ["handoff"], "status": "handoff",
            "handoff_reason": why, "sent_answer": HANDOFF_MESSAGE, "sent_by": "자동",
            "finished_at": now()}


def build_graph(checkpointer):
    g = StateGraph(State)
    for name, fn in [("classify", classify), ("retrieve", retrieve), ("answer", answer),
                     ("verify", verify), ("check", check), ("review", review),
                     ("send", send), ("rejected", rejected), ("handoff", handoff)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", gate_next, {"retrieve": "retrieve", "handoff": "handoff"})
    g.add_edge("retrieve", "answer")
    g.add_conditional_edges("answer", answer_next, {"verify": "verify", "handoff": "handoff"})
    g.add_conditional_edges("verify", verify_next, {"check": "check", "answer": "answer", "handoff": "handoff"})
    g.add_conditional_edges("check", check_next, {"review": "review", "send": "send"})
    g.add_conditional_edges("review", review_next, {"send": "send", "answer": "answer", "rejected": "rejected"})
    for n in ("send", "rejected", "handoff"):
        g.add_edge(n, END)
    return g.compile(checkpointer=checkpointer)


# ---------- 바깥에서 쓰는 함수 ----------
_app = None
_saver = None


def app():
    global _app, _saver
    if _app is None:
        STORE.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(STORE / "checkpoints.sqlite", check_same_thread=False)
        _saver = SqliteSaver(conn)
        _app = build_graph(_saver)
    return _app


def cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def submit(question: str, thread_id: Optional[str] = None) -> dict:
    """새 질문을 넣는다. 승인 기준에 걸리면 review에서 멈춘 채로 돌아온다."""
    thread_id = thread_id or "Q" + datetime.now().strftime("%m%d-%H%M%S")
    app().invoke({"question": question, "received_at": now()}, cfg(thread_id))
    return get(thread_id)


def answer_review(thread_id: str, action: str, text: str = "", reason: str = "",
                  instruction: str = "", by: str = "담당자") -> dict:
    """멈춘 건에 담당자 답을 넣어 멈춘 자리부터 이어서 실행한다."""
    d = {"action": action, "text": text, "reason": reason, "instruction": instruction, "by": by}
    app().invoke(Command(resume=d), cfg(thread_id))
    return get(thread_id)


def continue_run(thread_id: str) -> dict:
    """오류로 중간에 끊긴 건을 마지막 체크포인트부터 이어서 실행한다."""
    app().invoke(None, cfg(thread_id))
    return get(thread_id)


def get(thread_id: str) -> dict:
    snap = app().get_state(cfg(thread_id))
    v = dict(snap.values)
    v["thread_id"] = thread_id
    v["pending"] = bool(snap.next) and any(t.interrupts for t in snap.tasks)
    if v["pending"]:
        v["status"] = "pending"          # 승인 대기
    elif snap.next:
        v["status"] = "stuck"            # 오류로 중간에 끊김 (다음 단계가 남았는데 멈춘 것은 아님)
    return v


def thread_ids() -> list[str]:
    app()
    ids = []
    for cp in _saver.list(None):
        t = cp.config["configurable"]["thread_id"]
        if t not in ids:
            ids.append(t)
    return ids


def all_items() -> list[dict]:
    items = [get(t) for t in thread_ids()]
    return sorted(items, key=lambda x: x.get("received_at", ""))


def expire_pending(days: int = EXPIRE_DAYS) -> list[str]:
    """대기가 days일을 넘은 건은 자동 반려해 상담원 직접 처리로 넘긴다. (자동 승인은 하지 않는다)"""
    done = []
    limit = datetime.now() - timedelta(days=days)
    for it in all_items():
        if it["pending"] and datetime.fromisoformat(it["received_at"]) < limit:
            answer_review(it["thread_id"], "reject", reason=f"{days}일 동안 처리되지 않아 자동 반려", by="시스템")
            done.append(it["thread_id"])
    return done


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "25세 장애인 동생을 부양하는데, 기본공제와 장애인 공제를 받으려면 무엇이 필요한가요?"
    s = submit(q)
    print("thread:", s["thread_id"], "상태:", s["status"])
    print("카테고리:", s.get("categories"), "확신도:", s.get("confidence"))
    print("답변:", s.get("answer"))
    print("멈춘 이유:", [criteria.describe(k) for k in s.get("stop_reasons", [])])
