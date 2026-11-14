"""평가용 질문을 파이프라인에 넣고, 처리 결과를 집계한다.

사용법:  python run_batch.py            질문을 넣는다 (이미 넣은 건은 건너뛴다)
         python run_batch.py --summary  지금까지의 처리 결과를 results/summary.md 로 쓴다

질문은 data/goldenset.json 의 평가용(eval) 34건이고, thread_id 는 문항 번호(F01, M02, H03 …)다.
승인 기준에 걸린 건은 멈춘 채로 store/ 에 저장된다. 그 건은 데모 화면(app.py)에서 답을 준다.
중간에 오류로 끊겨도 다시 실행하면 끊긴 건부터 이어 간다.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import agent
import criteria
from prompts import GOLDEN

HERE = Path(__file__).parent
RESULTS = HERE / "results"

LABEL = {"pending": "승인 대기", "handoff": "상담원에게 넘김", "rejected": "반려", "stuck": "오류로 끊김"}
ACTION = {"approve": "승인", "edit": "수정 후 승인", "reject": "반려", "rejudge": "다시 판정"}


def outcome(it: dict) -> str:
    if it.get("status") == "sent":
        return {"자동": "자동 발송", "승인": "승인 후 발송", "수정 후 승인": "수정 후 발송"}[it["sent_by"]]
    return LABEL.get(it.get("status"), str(it.get("status")))


def eval_items() -> list[dict]:
    return [x for x in json.loads(GOLDEN.read_text(encoding="utf-8")) if x["split"] == "eval"]


def run():
    done = set(agent.thread_ids())
    for x in eval_items():
        t0 = time.time()
        if x["id"] in done:
            if agent.get(x["id"])["status"] != "stuck":
                print(f"  {x['id']} 이미 넣음 — 건너뜀")
                continue
            s = agent.continue_run(x["id"])
        else:
            s = agent.submit(x["question"], thread_id=x["id"])
        reasons = [criteria.short(k) for k in s.get("stop_reasons") or []]
        print(f"  {x['id']} {outcome(s)} {s.get('categories')} 확신도 {s.get('confidence')} "
              f"{round(time.time() - t0)}초 {reasons}")


def summary():
    RESULTS.mkdir(exist_ok=True)
    items = sorted(agent.all_items(), key=lambda x: x["thread_id"])
    answered = [it for it in items if it.get("stop_reasons") is not None]     # 답변까지 간 건
    stopped = [it for it in answered if it["stop_reasons"]]
    lines = [f"# 처리 결과 ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"- 모델: 판정 {agent.ROUTER_MODEL} / 답변 {agent.ANSWER_MODEL}",
             f"- 전체 {len(items)}건 = 답변까지 간 건 {len(answered)}건 + 답하지 않고 상담원에게 넘긴 건 {len(items) - len(answered)}건",
             f"- 답변까지 간 {len(answered)}건 중 승인 기준에 걸려 멈춘 건 {len(stopped)}건, 자동 발송 {len(answered) - len(stopped)}건", ""]
    lines += ["| 결과 | 건수 |", "|---|---|"]
    for k, n in Counter(outcome(it) for it in items).most_common():
        lines.append(f"| {k} | {n} |")
    lines += ["", "## 기준별 걸린 건수", "", "| 기준 | 건수 |", "|---|---|"]
    cnt = Counter(k for it in items for k in it.get("stop_reasons") or [])
    for k in criteria.SELECTED:
        lines.append(f"| {criteria.describe(k)} | {cnt.get(k, 0)} |")
    lines += ["", "## 건별", "", "| id | 카테고리 | 확신도 | 걸린 기준 | 결과 | 담당자 답 |", "|---|---|---|---|---|---|"]
    for it in items:
        ds = " → ".join(ACTION.get(d["action"], d["action"]) +
                        (f"({d.get('reason') or d.get('instruction')})" if d.get("reason") or d.get("instruction") else "")
                        for d in it.get("decisions", []))
        conf = it.get("confidence")
        lines.append(f"| {it['thread_id']} | {', '.join(it.get('categories', []))} | "
                     f"{'' if conf is None else f'{conf:.2f}'} | "
                     f"{', '.join(criteria.short(k) for k in it.get('stop_reasons') or []) or '-'} | {outcome(it)} | {ds or '-'} |")
    lines += ["", "## 답변", ""]
    for it in items:
        lines += [f"### {it['thread_id']} {it['question']}", "",
                  f"- AI 답변: {it.get('answer') or '-'}",
                  f"- 고객에게 나간 글: {it.get('sent_answer', '-') if it.get('status') in ('sent', 'handoff') else '-'}", ""]
    out = RESULTS / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    keep = [{k: v for k, v in it.items() if k != "context"} for it in items]      # 근거 본문은 길어서 뺀다
    (RESULTS / "summary.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run()
