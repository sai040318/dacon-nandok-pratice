# kor_obfus_preprocess.py
# -*- coding: utf-8 -*-
"""
kor_obfus_preprocess.py

난독화된 한글 리뷰 복원용 전처리 + 룰 기반 복원 유틸 모듈.
(EDA/그래프/데이터프레임 출력 없음)

기능:
- 한글 음절 분해/조합: (초성 L, 중성 V, 종성 T)
- input/output 정렬 (공백 제거 후 길이 맞추기)
- train (input, output) 쌍에서 자모 변화 통계 수집
- 자주 등장하는 자모 치환 룰 생성
- 룰을 적용해 난독화 텍스트 복원
- train 데이터에서 input/pred_rule/output 를 간단히 출력해 보는 함수
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

# pandas는 선택 사항이지만, show_rule_application에서 DataFrame을 가정함
import pandas as pd  # type: ignore

# -------------------------
# 기본 텍스트 전처리
# -------------------------

_WS_RE = re.compile(r"\s+")
_ZWSP_RE = re.compile("\u200b")


def normalize_text(s: Union[str, Any]) -> str:
    """제로폭 공백 제거, 연속 공백을 하나로, 앞뒤 공백 제거."""
    if not isinstance(s, str):
        s = str(s)
    s = _ZWSP_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def remove_spaces(s: Union[str, Any]) -> str:
    """모든 공백 문자 제거."""
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", "", s)


# -------------------------
# 한글 음절 <-> 자모
# -------------------------

S_BASE = 0xAC00
L_COUNT = 19
V_COUNT = 21
T_COUNT = 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT

CHOSUNG_LIST: List[str] = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNGSUNG_LIST: List[str] = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONGSUNG_LIST: List[str] = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ",
    "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]


def is_hangul_syllable(ch: str) -> bool:
    """'가'..'힣' 영역의 한글 음절인지 여부."""
    if not ch or len(ch) != 1:
        return False
    code = ord(ch)
    return S_BASE <= code <= (S_BASE + S_COUNT - 1)


def decompose_char(ch: str) -> Optional[Tuple[str, str, str]]:
    """한글 음절을 (초성 L, 중성 V, 종성 T)로 분해. 한글이 아니면 None."""
    if not is_hangul_syllable(ch):
        return None
    code = ord(ch) - S_BASE
    l_index = code // N_COUNT
    v_index = (code % N_COUNT) // T_COUNT
    t_index = code % T_COUNT
    return (
        CHOSUNG_LIST[l_index],
        JUNGSUNG_LIST[v_index],
        JONGSUNG_LIST[t_index],
    )


def compose_char(L: str, V: str, T: str = "") -> str:
    """(L, V, T) 자모를 한글 음절로 합성. 잘못된 자모면 '' 반환."""
    try:
        l_idx = CHOSUNG_LIST.index(L)
        v_idx = JUNGSUNG_LIST.index(V)
        t_idx = JONGSUNG_LIST.index(T)
    except ValueError:
        return ""
    return chr(S_BASE + l_idx * N_COUNT + v_idx * T_COUNT + t_idx)


# -------------------------
# input/output 정렬 (자모 통계용)
# -------------------------

def align_text_pair_for_jamo(inp: Union[str, Any],
                             out: Union[str, Any]) -> Tuple[str, str]:
    """
    자모 통계 계산을 위해 input/output을 정렬:
      - normalize_text
      - 공백 제거
      - 둘 중 더 짧은 길이에 맞춰 잘라냄
    """
    a = remove_spaces(normalize_text(inp))
    b = remove_spaces(normalize_text(out))
    L = min(len(a), len(b))
    return a[:L], b[:L]


# -------------------------
# 자모 변화 통계
# -------------------------

@dataclass(frozen=True)
class JamoStats:
    # src != dst 인 경우의 변화만 기록
    L_changes: Counter  # (src_L, dst_L) -> count
    V_changes: Counter  # (src_V, dst_V) -> count
    T_changes: Counter  # (src_T, dst_T) -> count

    # input에서 자모 등장 횟수 (src 기준)
    L_totals: Counter   # src_L -> count
    V_totals: Counter   # src_V -> count
    T_totals: Counter   # src_T -> count

    # 전체 비교한 문자 수, 그 중 한글 쌍 수
    total_pairs: int
    hangul_pairs: int


def _iter_pairs(
    df_or_pairs: Any,
    input_col: str,
    output_col: str,
    sample_n: Optional[int],
    random_state: int,
) -> Iterator[Tuple[str, str]]:
    """
    - DataFrame이면 input_col, output_col에서 가져오고,
    - 아니면 (input, output) iterable이라고 보고 그대로 사용.
    """
    if pd is not None and hasattr(df_or_pairs, "iloc") and hasattr(df_or_pairs, "columns"):
        df = df_or_pairs
        if sample_n is not None:
            df = df.sample(sample_n, random_state=random_state)
        return ((str(i), str(o)) for i, o in zip(df[input_col], df[output_col]))
    # iterable of pairs
    return ((str(i), str(o)) for i, o in df_or_pairs)


def collect_jamo_change_stats(
    df_or_pairs: Any,
    *,
    input_col: str = "input",
    output_col: str = "output",
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> JamoStats:
    """
    (input, output) 쌍들로부터 자모 변화 통계 수집.

    - input 기준으로 각 자모가 몇 번 나왔는지 (L/V/T_totals)
    - src != dst 인 (src, dst) 변화 카운트 (L/V/T_changes)
    """
    L_changes, V_changes, T_changes = Counter(), Counter(), Counter()
    L_totals, V_totals, T_totals = Counter(), Counter(), Counter()

    total_pairs = 0
    hangul_pairs = 0

    for inp, out in _iter_pairs(df_or_pairs, input_col, output_col, sample_n, random_state):
        inp_aligned, out_aligned = align_text_pair_for_jamo(inp, out)
        total_pairs += len(inp_aligned)

        for ch_in, ch_out in zip(inp_aligned, out_aligned):
            d_in = decompose_char(ch_in)
            d_out = decompose_char(ch_out)
            if d_in is None or d_out is None:
                continue

            hangul_pairs += 1
            Li, Vi, Ti = d_in
            Lo, Vo, To = d_out

            L_totals[Li] += 1
            V_totals[Vi] += 1
            T_totals[Ti] += 1

            if Li != Lo:
                L_changes[(Li, Lo)] += 1
            if Vi != Vo:
                V_changes[(Vi, Vo)] += 1
            if Ti != To:
                T_changes[(Ti, To)] += 1

    return JamoStats(
        L_changes=L_changes,
        V_changes=V_changes,
        T_changes=T_changes,
        L_totals=L_totals,
        V_totals=V_totals,
        T_totals=T_totals,
        total_pairs=total_pairs,
        hangul_pairs=hangul_pairs,
    )


# -------------------------
# 룰 생성 & 복원
# -------------------------

def build_axis_rules(
    changes: Counter,
    totals: Counter,
    *,
    min_total: int = 50,
    min_top_prob: float = 0.8,
    allow_identity: bool = False,
) -> Dict[str, str]:
    """
    한 축(L/V/T)에 대한 자모 치환 룰 생성.

    각 src에 대해:
      - totals[src] >= min_total 일 때만 후보
      - changes + identity(src->src)를 합쳐 분포를 만들고
      - argmax dst를 구한 뒤, top_prob >= min_top_prob 이면 룰로 채택
      - top_dst == src 이고 allow_identity=False 이면 저장하지 않음 (그냥 그대로 두면 됨)
    """
    rules: Dict[str, str] = {}
    for src, total in totals.items():
        total = int(total)
        if total < min_total or total <= 0:
            continue

        # src에서 dst로 변한 경우만 모으기
        dst_cnt: Dict[str, int] = {}
        for (s, dst), cnt in changes.items():
            if s == src:
                dst_cnt[dst] = dst_cnt.get(dst, 0) + int(cnt)

        changed_sum = sum(dst_cnt.values())
        same_cnt = total - changed_sum  # src->src (identity)

        candidates: List[Tuple[int, str]] = []
        if same_cnt > 0:
            candidates.append((same_cnt, src))
        for dst, cnt in dst_cnt.items():
            candidates.append((cnt, dst))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_cnt, top_dst = candidates[0]
        top_prob = top_cnt / total

        if top_dst == src and not allow_identity:
            # 그냥 그대로 두는 게 최선인 경우 -> 룰 저장 X
            continue
        if top_prob >= min_top_prob:
            rules[src] = top_dst

    return rules


@dataclass(frozen=True)
class JamoRules:
    L_rule: Dict[str, str]
    V_rule: Dict[str, str]
    T_rule: Dict[str, str]


def build_rules_from_train(
    df_or_pairs: Any,
    *,
    min_total: int = 50,
    min_top_prob: float = 0.8,
    allow_identity: bool = False,
    input_col: str = "input",
    output_col: str = "output",
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[JamoRules, JamoStats]:
    """
    통계 수집 + 룰 생성까지 한 번에.

    반환:
      - JamoRules(L_rule, V_rule, T_rule)
      - JamoStats (참고용)
    """
    stats = collect_jamo_change_stats(
        df_or_pairs,
        input_col=input_col,
        output_col=output_col,
        sample_n=sample_n,
        random_state=random_state,
    )
    L_rule = build_axis_rules(
        stats.L_changes, stats.L_totals,
        min_total=min_total, min_top_prob=min_top_prob, allow_identity=allow_identity
    )
    V_rule = build_axis_rules(
        stats.V_changes, stats.V_totals,
        min_total=min_total, min_top_prob=min_top_prob, allow_identity=allow_identity
    )
    T_rule = build_axis_rules(
        stats.T_changes, stats.T_totals,
        min_total=min_total, min_top_prob=min_top_prob, allow_identity=allow_identity
    )
    return JamoRules(L_rule=L_rule, V_rule=V_rule, T_rule=T_rule), stats


def restore_with_jamo_rules(text: Union[str, Any], rules: JamoRules) -> str:
    """
    텍스트의 각 한글 음절에 대해 자모 분해 → 축별 룰 적용 → 재조합.
    한글이 아닌 문자는 그대로 둔다.
    """
    if not isinstance(text, str):
        text = str(text)

    out_chars: List[str] = []
    for ch in text:
        d = decompose_char(ch)
        if d is None:
            out_chars.append(ch)
            continue

        L, V, T = d
        L2 = rules.L_rule.get(L, L)
        V2 = rules.V_rule.get(V, V)
        T2 = rules.T_rule.get(T, T)
        new_ch = compose_char(L2, V2, T2)
        out_chars.append(new_ch if new_ch else ch)

    return "".join(out_chars)


# -------------------------
# 편의 함수: train에서 input/pred/output 보기
# -------------------------

def show_rule_application(
    df: "pd.DataFrame",
    rules: JamoRules,
    *,
    n: int = 5,
    seed: int = 42,
    id_col: str = "ID",
    input_col: str = "input",

    output_col: str = "output",
) -> None:
    """
    DataFrame(df)에서 n개 샘플을 뽑아
      - [INPUT ]
      - [PRED  ] (룰 적용 결과)
      - [OUTPUT]
    을 출력해주는 함수.

    ex)
      from kor_obfus_preprocess import build_rules_from_train, restore_with_jamo_rules, show_rule_application
      rules, stats = build_rules_from_train(train)
      show_rule_application(train, rules, n=5)
    """
    if pd is None:
        raise ImportError("pandas가 필요합니다. `pip install pandas` 후 사용해주세요.")

    import random
    random.seed(seed)

    if n <= 0:
        return

    if len(df) == 0:
        print("DataFrame is empty.")
        return

    idx_list = list(range(len(df)))
    if n > len(idx_list):
        n = len(idx_list)
    sample_idx = random.sample(idx_list, n)

    for i in sample_idx:
        row = df.iloc[i]
        inp = str(row[input_col])
        out = str(row[output_col])
        pred = restore_with_jamo_rules(inp, rules)

        print("=" * 100)
        if id_col in df.columns:
            print(f"📌 ID: {row[id_col]}")
        else:
            print(f"📌 index: {i}")
        print(f"[INPUT ] {inp}")
        print(f"[PRED  ] {pred}")
        print(f"[OUTPUT] {out}")


__all__ = [
    "normalize_text",
    "remove_spaces",
    "is_hangul_syllable",
    "decompose_char",
    "compose_char",
    "align_text_pair_for_jamo",
    "JamoStats",
    "collect_jamo_change_stats",
    "build_axis_rules",
    "JamoRules",
    "build_rules_from_train",
    "restore_with_jamo_rules",
    "show_rule_application",
    "CHOSUNG_LIST",
    "JUNGSUNG_LIST",
    "JONGSUNG_LIST",
]