"""승인 기준.

답변이 만들어진 뒤, 고객에게 보내기 전에 계산한다. LLM을 쓰지 않고 코드로만 계산하므로 같은 건은 항상 같은 결과가 나온다.
기준은 "모델이 틀릴 것 같은 건"이 아니라 "틀렸을 때 손해가 큰 건"을 고른다.

- CANDIDATES: 비교해 본 기준 후보 전부 (validate.py가 개입·놓침·헛멈춤을 센다)
- SELECTED:   파이프라인이 실제로 쓰는 기준 (agent.py의 check 단계)
"""
from __future__ import annotations

import re

APPROVAL_CONF = 0.7   # 확신도 기준 (비교용 후보)
NEAR = 0.25           # 기준선 근처: 질문의 금액이 답변에 나온 기준 금액의 ±25% 안

_UNIT = {"억": 10**8, "천만": 10**7, "백만": 10**6, "만": 10**4, "천": 10**3, None: 1}
_AMOUNT = re.compile(r"(\d[\d,]*)\s*(억|천만|백만|만|천)?\s*원")
_MONTHLY = re.compile(r"(매달|매월|달마다|한\s*달에|월)\s*$")


def amounts(text: str) -> set[int]:
    """글에 나온 금액을 원 단위로 모은다. 금액 바로 앞에 '매달·매월'이 있으면 12를 곱한 연 금액도 넣는다."""
    found = set()
    for m in _AMOUNT.finditer(text):
        won = int(m.group(1).replace(",", "")) * _UNIT[m.group(2)]
        if won <= 0:
            continue
        found.add(won)
        if _MONTHLY.search(text[:m.start()]):
            found.add(won * 12)
    return found


def near_pairs(case: dict, width: float = NEAR) -> list[tuple[int, int]]:
    """(질문의 금액, 답변에 나온 기준 금액) 가운데 서로 width 비율 안에 있는 쌍."""
    asked = amounts(case["question"])
    lines = amounts(case["answer"]) - asked          # 답변이 질문 금액을 되풀이한 것은 기준 금액이 아니다
    return sorted((q, t) for q in asked for t in lines if abs(q / t - 1) <= width)


# ---------- 기준 하나하나: case → 멈출지(True/False) ----------
def low_conf(case: dict) -> bool:
    return case["confidence"] < APPROVAL_CONF


def multi(case: dict) -> bool:
    return len(case["categories"]) >= 2


def disability(case: dict) -> bool:
    return "장애" in case["question"] + case["answer"]


def has_amount(case: dict) -> bool:
    return bool(amounts(case["answer"]))


def asks_amount(case: dict) -> bool:
    return bool(amounts(case["question"]))


_OTHER_YEAR = re.compile(r"(?<!\d)(?!2025)20\d\d\s*년|작년|지난해|재작년|내년")


def other_year(case: dict) -> bool:
    return bool(_OTHER_YEAR.search(case["question"]))


def near_line(case: dict) -> bool:
    return bool(near_pairs(case))


def near_line_narrow(case: dict) -> bool:
    return bool(near_pairs(case, 0.10))


def rewritten(case: dict) -> bool:
    return case.get("attempts", 1) >= 2


def rejudged(case: dict) -> bool:
    return case.get("rejudges", 0) > 0


# 키: (짧은 이름, 화면·보고서에 쓰는 설명, 대비하는 위험, 계산 함수)
CANDIDATES = {
    "low_conf": ("확신도 낮음", f"분류 확신도가 {APPROVAL_CONF} 미만",
                 "카테고리를 잘못 골라 엉뚱한 규칙으로 답함", low_conf),
    "multi": ("두 카테고리", "두 카테고리에 걸친 질문",
              "두 규칙을 합쳐 판단하다가 요건 하나를 빠뜨림", multi),
    "disability": ("장애", "질문이나 답변에 '장애'가 들어감",
                   "장애인 공제는 장애인증명서 같은 서류 요건이 있는데 이를 빠뜨리면 고객이 공제를 잘못 신청함", disability),
    "has_amount": ("금액 있음", "답변에 금액이 들어감",
                   "금액을 잘못 안내함", has_amount),
    "asks_amount": ("질문에 금액", "질문에 금액이 들어감",
                    "고객이 금액을 대며 물으면 답이 금액 비교에 달려 있어, 비교를 틀리거나 빼먹으면 '공제 가능'과 '불가'가 뒤바뀜", asks_amount),
    "other_year": ("다른 연도", "질문에 2025년이 아닌 연도(작년·내년 포함)가 들어감",
                   "자료가 2025년 귀속 기준이라 다른 연도의 일을 물으면 어느 해 연말정산에 적용되는지 혼동함", other_year),
    "near_line": ("기준선 근처", f"질문의 금액이 답변에 나온 기준 금액의 ±{int(NEAR * 100)}% 안",
                  "금액 크기를 잘못 비교해 '공제 가능'과 '불가'가 뒤바뀜", near_line),
    "near_line_narrow": ("기준선 근처(좁게)", "질문의 금액이 답변에 나온 기준 금액의 ±10% 안",
                         "금액 크기를 잘못 비교해 '공제 가능'과 '불가'가 뒤바뀜", near_line_narrow),
    "rewritten": ("다시 쓴 답", "검증에 걸려 다시 쓴 답",
                  "근거에 없는 숫자나 근거 번호를 한 번 썼던 답이라 다른 부분도 틀렸을 수 있음", rewritten),
    "rejudge": ("다시 판정", "담당자가 다시 판정을 요청해 새로 쓴 답",
                "새로 쓴 답을 아무도 확인하지 않고 내보냄", rejudged),
}

# 파이프라인이 실제로 쓰는 기준. 고른 까닭은 REPORT.md 3절.
SELECTED = ["multi", "disability", "asks_amount", "other_year", "rejudge"]


def short(key: str) -> str:
    return CANDIDATES[key][0]


def describe(key: str) -> str:
    return CANDIDATES[key][1]


def risk(key: str) -> str:
    return CANDIDATES[key][2]


def stop_reasons(case: dict, keys: list[str] | None = None) -> list[str]:
    """걸린 기준의 키 목록. 비어 있으면 사람을 거치지 않고 자동 발송한다."""
    return [k for k in (SELECTED if keys is None else keys) if CANDIDATES[k][3](case)]
