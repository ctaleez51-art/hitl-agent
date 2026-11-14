"""승인 기준 검증: 기준 후보마다 개입·놓침·헛멈춤을 센다.

run_batch.py 로 질문을 넣은 뒤에 실행한다. 담당자가 답을 주기 전의 AI 답변을 채점한다.

- 사람이 봐야 하는 건 = 채점에서 틀린 답 (must 를 하나라도 빠뜨렸거나 forbid 를 주장한 답)
- 개입   = 그 기준으로 멈춘 건 / 답변까지 간 건
- 놓침   = 틀린 답인데 멈추지 않아 자동 발송되는 건
- 헛멈춤 = 맞는 답인데 멈춘 건

사용법:  python validate.py                기준 비교표를 results/criteria.md 로 쓴다
         python validate.py --judge-check  채점기 검증만 (모범 답안은 통과, forbid 오답은 탈락해야 한다)
         python validate.py --fresh        저장해 둔 채점 결과를 버리고 다시 채점한다

한 번 채점한 건은 results/judged.json 에 저장해 두고 다시 채점하지 않는다.
(담당자가 "다시 판정"으로 답을 바꾼 뒤에 실행해도 처음 답변의 채점이 유지된다.)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

import agent
import criteria
from prompts import JUDGE_SYSTEM
from run_batch import eval_items

HERE = Path(__file__).parent
RESULTS = HERE / "results"
JUDGED = RESULTS / "judged.json"
JUDGE_MODEL = agent.os.environ.get("JUDGE_MODEL", agent.ANSWER_MODEL)

# 비교할 기준 (이름, 기준 키 목록). 목록 중 하나라도 걸리면 멈춘다.
ROWS = [
    ("기준 없음 (전부 자동)", []),
    ("확신도 0.7 미만", ["low_conf"]),
    ("답변에 금액이 들어감", ["has_amount"]),
    ("검증에 걸려 다시 쓴 답", ["rewritten"]),
    ("두 카테고리", ["multi"]),
    ("'장애' 포함", ["disability"]),
    ("다른 연도", ["other_year"]),
    ("기준선 근처 ±10%", ["near_line_narrow"]),
    ("기준선 근처 ±25%", ["near_line"]),
    ("질문에 금액", ["asks_amount"]),
    ("두 카테고리 + '장애'", ["multi", "disability"]),
    ("두 카테고리 + '장애' + 기준선 근처 ±25%", ["multi", "disability", "near_line"]),
    ("두 카테고리 + '장애' + 기준선 근처 ±25% + 다시 쓴 답", ["multi", "disability", "near_line", "rewritten"]),
    ("두 카테고리 + '장애' + 질문에 금액", ["multi", "disability", "asks_amount"]),
    ("두 카테고리 + '장애' + 질문에 금액 + 다시 쓴 답", ["multi", "disability", "asks_amount", "rewritten"]),
    ("두 카테고리 + '장애' + 질문에 금액 + 다른 연도", ["multi", "disability", "asks_amount", "other_year"]),
    ("두 카테고리 + '장애' + 질문에 금액 + 다른 연도 + 다시 쓴 답", ["multi", "disability", "asks_amount", "other_year", "rewritten"]),
    ("전부 멈춤", None),
]


class Verdict(BaseModel):
    must_ok: list[bool] = Field(description="must 항목마다 충족 여부, 순서대로")
    forbid_hit: list[bool] = Field(description="forbid 항목마다 위반 여부, 순서대로")
    passed: bool
    reason: str = Field(description="한두 문장")


_judge = None


def judge(item: dict, answer_text: str) -> Verdict:
    global _judge
    if _judge is None:
        _judge = init_chat_model(JUDGE_MODEL, model_provider="openai", temperature=0,
                                 timeout=60, max_retries=2).with_structured_output(Verdict)
    body = json.dumps({"question": item["question"], "must": item["must"],
                       "forbid": item["forbid"], "answer": answer_text}, ensure_ascii=False, indent=1)
    v: Verdict = _judge.invoke([("system", JUDGE_SYSTEM), ("human", body)])
    # passed 는 모델 말이 아니라 항목별 판정으로 다시 계산한다
    v.passed = all(v.must_ok) and len(v.must_ok) == len(item["must"]) and not any(v.forbid_hit)
    return v


def judge_check(items: list[dict]) -> bool:
    ok_ref = ok_bad = n = 0
    for it in items:
        if not it.get("reference"):
            continue
        n += 1
        good = judge(it, it["reference"])
        bad = judge(it, it["forbid"][0] + ".")
        ok_ref += good.passed
        ok_bad += not bad.passed
        print(f"  {it['id']}: 모범답안 {'통과' if good.passed else '탈락(문제)'} / 오답 {'탈락' if not bad.passed else '통과(문제)'}")
    print(f"→ 모범 답안 통과 {ok_ref}/{n}, forbid 오답 탈락 {ok_bad}/{n}")
    return ok_ref == n and ok_bad == n


def collect(fresh: bool) -> list[dict]:
    """저장소에서 답변까지 간 건을 꺼내 채점하고, 후보 기준마다 걸리는지 계산한다."""
    saved = {} if fresh or not JUDGED.exists() else json.loads(JUDGED.read_text(encoding="utf-8"))
    ids = set(agent.thread_ids())
    cases = []
    for it in eval_items():
        if it["id"] not in ids:
            print(f"  {it['id']} 저장소에 없음 — run_batch.py 를 먼저 실행")
            continue
        s = agent.get(it["id"])
        if s.get("stop_reasons") is None:            # 답변까지 가지 않고 넘긴 건은 발송되는 답이 없다
            continue
        if it["id"] not in saved:
            v = judge(it, s["answer"])
            saved[it["id"]] = {"answer": s["answer"], "passed": v.passed, "reason": v.reason,
                               "categories": s["categories"], "confidence": s["confidence"],
                               "attempts": s["attempts"], "question": it["question"],
                               "hard": it.get("hard", "")}
            print(f"  {it['id']} 채점 {'O' if v.passed else 'X'}")
        c = saved[it["id"]]
        c["hits"] = [k for k in criteria.CANDIDATES if k != "rejudge" and criteria.CANDIDATES[k][3](c)]
        cases.append({"id": it["id"], **c})
    RESULTS.mkdir(exist_ok=True)
    JUDGED.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    return cases


def measure(cases: list[dict], keys: list[str] | None) -> tuple[int, int, int]:
    stopped = missed = false_stop = 0
    for c in cases:
        did_stop = True if keys is None else any(k in c["hits"] for k in keys)
        if did_stop:
            stopped += 1
            false_stop += c["passed"]
        elif not c["passed"]:
            missed += 1
    return stopped, missed, false_stop


def write(cases: list[dict]) -> Path:
    n, wrong = len(cases), [c for c in cases if not c["passed"]]
    L = [f"# 승인 기준 비교 ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"- 모델: 판정 {agent.ROUTER_MODEL} / 답변 {agent.ANSWER_MODEL} / 채점 {JUDGE_MODEL}",
         f"- 답변까지 간 건 {n}건, 그중 채점에서 틀린 답(사람이 봐야 하는 건) {len(wrong)}건: {', '.join(c['id'] for c in wrong) or '없음'}",
         "- 놓침 = 틀린 답인데 멈추지 않는 건, 헛멈춤 = 맞는 답인데 멈춘 건", "",
         "| 멈춤 기준 | 개입 | 놓침 | 헛멈춤 |", "|---|---|---|---|"]
    for name, keys in ROWS:
        s, m, f = measure(cases, keys)
        mark = " ← 쓰는 기준" if keys is not None and sorted(keys) == sorted(k for k in criteria.SELECTED if k != "rejudge") else ""
        L.append(f"| {name}{mark} | {s}/{n} | {m} | {f} |")
    used = [k for k in criteria.SELECTED if k != "rejudge"]
    L += ["", "## 쓰는 기준을 문항 묶음별로 보면", "", "| 문항 | 개입 | 놓침 | 헛멈춤 |", "|---|---|---|---|"]
    for name, part in [("라우팅 과제에서 가져온 문항", [c for c in cases if not c["hard"]]),
                       ("이번에 더한 어려운 문항", [c for c in cases if c["hard"]])]:
        s, m, f = measure(part, used)
        L.append(f"| {name} | {s}/{len(part)} | {m} | {f} |")
    L += ["", "## 건별", "", "| id | 어려운 점 | 채점 | 걸린 후보 기준 | 채점 이유 |", "|---|---|---|---|---|"]
    for c in cases:
        hits = ", ".join(criteria.short(k) for k in c["hits"]) or "-"
        L.append(f"| {c['id']} | {c['hard'] or '-'} | {'O' if c['passed'] else 'X'} | {hits} | {c['reason'].replace('|', '/')} |")
    L += ["", "## 틀린 답", ""]
    for c in wrong:
        L += [f"### {c['id']} {c['question']}", "", f"- AI 답변: {c['answer']}", f"- 채점: {c['reason']}", ""]
    out = RESULTS / "criteria.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-check", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.judge_check:
        judge_check(eval_items())
    else:
        cases = collect(a.fresh)
        print(f"→ {write(cases)}")
        for name, keys in ROWS:
            print(f"  {name}: 개입 {measure(cases, keys)[0]}/{len(cases)} 놓침 {measure(cases, keys)[1]} 헛멈춤 {measure(cases, keys)[2]}")
