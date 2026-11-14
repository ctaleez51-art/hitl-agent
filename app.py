"""승인 데모 화면.  실행: streamlit run app.py"""
from datetime import datetime

import streamlit as st

import agent
import context as ctx
import criteria
from run_batch import ACTION, outcome

st.set_page_config(page_title="인적공제 답변 승인", page_icon="🧾", layout="centered")
st.title("인적공제 상담 답변 승인")
st.caption(f"승인 기준에 걸린 답만 담당자가 보고, 나머지는 자동으로 발송됩니다. "
           f"{agent.EXPIRE_DAYS}일 넘게 대기한 건은 자동 반려되어 상담원이 직접 처리합니다. "
           f"(모델: 판정 {agent.ROUTER_MODEL} / 답변 {agent.ANSWER_MODEL})")

expired = agent.expire_pending()
if expired:
    st.warning(f"{agent.EXPIRE_DAYS}일 넘게 대기해 자동 반려된 건: {', '.join(expired)}")

# ---------- 새 질문 넣기 ----------
with st.sidebar:
    st.header("새 질문 넣기")
    examples = ["25세 장애인 동생을 부양하는데, 기본공제와 장애인 공제를 받으려면 무엇이 필요한가요?",
                "아버지가 국민연금을 매달 45만원씩 받으시는데 기본공제 되나요?",
                "결혼식은 안 하고 혼인신고만 했는데 배우자 공제 되나요?",
                "종합소득세 신고 기한이 언제예요?"]
    pick = st.selectbox("예시 질문", ["(직접 입력)"] + examples)
    q = st.text_area("고객 질문", value="" if pick == "(직접 입력)" else pick)
    if st.button("에이전트에 넣기", type="primary") and q.strip():
        with st.spinner("판정 → 근거 조회 → 답변 → 검증 → 승인 기준 계산 중..."):
            s = agent.submit(q.strip())
        st.session_state["flash"] = f"{s['thread_id']}: {outcome(s)}"
        st.rerun()
    st.divider()
    st.subheader("승인 기준")
    st.caption("하나라도 걸리면 고객에게 보내기 전에 멈춥니다.")
    for k in criteria.SELECTED:
        st.write(f"- {criteria.describe(k)}")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

items = agent.all_items()
pending = [it for it in items if it["pending"]]
finished = [it for it in items if not it["pending"]]


def waited(it: dict) -> str:
    sec = (datetime.now() - datetime.fromisoformat(it["received_at"])).total_seconds()
    return f"{int(sec // 86400)}일" if sec >= 86400 else f"{int(sec // 3600)}시간" if sec >= 3600 else f"{int(sec // 60)}분"


def decision_line(d: dict) -> str:
    note = d.get("instruction") or d.get("reason") or ""
    return f"{d['at']} · {d.get('by', '담당자')} · {ACTION.get(d['action'], d['action'])}" + (f" · {note}" if note else "")


tab1, tab2 = st.tabs([f"승인 대기 ({len(pending)})", f"처리 완료 ({len(finished)})"])

# ---------- 승인 대기 ----------
with tab1:
    if not pending:
        st.info("승인을 기다리는 건이 없습니다.")
    else:
        labels = {f"{it['thread_id']} · {it['question'][:30]}… · 대기 {waited(it)}": it for it in pending}
        it = labels[st.radio("대기 목록 (한 건을 고르면 아래에 열립니다)", list(labels))]
        st.divider()

        st.markdown("**① 고객 질문**")
        st.write(it["question"])

        st.markdown("**② AI 답변** (승인하면 이 글이 고객에게 나갑니다)")
        st.info(it["answer"])

        st.markdown("**③ AI 판정과 근거**")
        st.write(f"카테고리 {', '.join(it['categories'])} · 확신도 {it['confidence']:.2f} — {it['reason']}")
        cited = it["checks"][-1]["cited"] if it.get("checks") else []
        for cid in cited:
            if cid in ctx.ALL:
                with st.expander(f"근거로 삼은 자료 {cid} · {ctx.ALL[cid].title}"):
                    st.write(ctx.ALL[cid].text.strip())

        st.markdown("**④ 멈춘 이유**")
        st.warning("\n".join(f"- **{criteria.describe(k)}** — 대비하는 위험: {criteria.risk(k)}" for k in it["stop_reasons"]))

        st.markdown("**⑤ 승인하면 일어나는 일**")
        st.error("위 AI 답변이 고객에게 그대로 발송됩니다. 발송한 뒤에는 되돌릴 수 없습니다.")

        if it.get("decisions"):
            with st.expander("이 건에 앞서 준 담당자 답"):
                for d in it["decisions"]:
                    st.write(decision_line(d))

        st.divider()
        left = agent.MAX_REJUDGE - it.get("rejudges", 0)
        options = {"승인": "approve", "수정 후 승인": "edit", "반려": "reject",
                   f"다시 판정 (남은 횟수 {left})": "rejudge"}
        choice = options[st.radio("담당자 답", list(options), horizontal=True, key=f"c-{it['thread_id']}")]
        text = reason = instruction = ""
        if choice == "approve":
            st.caption("AI 답변을 그대로 고객에게 보냅니다.")
        elif choice == "edit":
            text = st.text_area("고객에게 보낼 답 (고쳐 쓰기)", value=it["answer"], height=160, key=f"t-{it['thread_id']}")
        elif choice == "reject":
            reason = st.text_input("반려 사유 (필수) — 고객에게 보내지 않고 상담원이 직접 처리합니다", key=f"r-{it['thread_id']}")
        elif choice == "rejudge":
            instruction = st.text_input("AI에게 줄 지시 (필수) 예: 장애인증명서 요건도 함께 쓰세요", key=f"i-{it['thread_id']}")
            st.caption("AI가 지시를 받아 답을 다시 쓰고, 새 답은 다시 승인 대기로 옵니다.")

        ok = {"approve": True, "edit": bool(text.strip()), "reject": bool(reason.strip()),
              "rejudge": bool(instruction.strip()) and left > 0}[choice]
        if st.button("답 보내기", type="primary", disabled=not ok):
            with st.spinner("멈춘 자리부터 이어서 실행 중..."):
                s = agent.answer_review(it["thread_id"], choice, text=text.strip(),
                                        reason=reason.strip(), instruction=instruction.strip())
            st.session_state["flash"] = f"{s['thread_id']}: {outcome(s)}"
            st.rerun()

# ---------- 처리 완료 ----------
with tab2:
    if not finished:
        st.info("처리된 건이 없습니다.")
    else:
        st.dataframe([{"id": it["thread_id"], "질문": it["question"], "결과": outcome(it),
                       "걸린 기준": ", ".join(criteria.short(k) for k in it.get("stop_reasons") or []) or "-"}
                      for it in finished], hide_index=True, width="stretch")
        for it in finished:
            with st.expander(f"{it['thread_id']} · {outcome(it)} · {it['question'][:30]}…"):
                if it.get("status") in ("sent", "handoff"):
                    st.write("고객에게 나간 글:", it.get("sent_answer"))
                if it.get("status") == "handoff":
                    st.write("넘긴 이유:", it.get("handoff_reason"))
                if it.get("status") == "rejected":
                    st.write("AI 답변(발송 안 함):", it.get("answer"))
                    st.write("반려 사유:", it.get("reject_reason"))
                for d in it.get("decisions", []):
                    st.write("담당자 답:", decision_line(d))
