import unicodedata
import pandas as pd
from collections import Counter

# ---------------------------
# 1) 자모 분해
# ---------------------------
def decompose_text(text: str):
    return [unicodedata.normalize("NFD", ch) for ch in text]


# ---------------------------
# 2) 인풋 vs 아웃풋 차이 계산
# ---------------------------
def get_differences(inp_dec, out_dec):
    diffs = []
    min_len = min(len(inp_dec), len(out_dec))

    for i in range(min_len):
        if inp_dec[i] != out_dec[i]:
            diffs.append((inp_dec[i], out_dec[i]))

    if len(inp_dec) != len(out_dec):
        diffs.append(("len_diff", f"{len(inp_dec)} vs {len(out_dec)}"))

    return diffs


# ---------------------------
# 3) 전체 train 기반 규칙 생성
# ---------------------------
def build_rule_dict(df: pd.DataFrame, threshold=100):
    diff_counter = Counter()

    for inp, out in zip(df["input"], df["output"]):
        inp_dec = decompose_text(inp)
        out_dec = decompose_text(out)
        diffs = get_differences(inp_dec, out_dec)

        for d in diffs:
            diff_counter[d] += 1

    # threshold 이상 발생한 변환만 규칙으로 채택
    rule_dict = {
        inp_jamo: out_jamo
        for (inp_jamo, out_jamo), cnt in diff_counter.items()
        if inp_jamo != "len_diff" and cnt >= threshold
    }

    return rule_dict


# ---------------------------
# 4) 규칙 적용 함수
# ---------------------------
def apply_rules(text: str, rule_dict):
    result = []
    for ch in text:
        dec = unicodedata.normalize("NFD", ch)
        if dec in rule_dict:
            dec = rule_dict[dec]
        result.append(unicodedata.normalize("NFC", dec))
    return "".join(result)


# ---------------------------
# 5) DataFrame 전체에 규칙 적용
# ---------------------------
def preprocess_dataframe(df: pd.DataFrame, rule_dict):
    df = df.copy()
    df["input_cleaned"] = df["input"].apply(lambda x: apply_rules(x, rule_dict))
    return df
