# dacon-nandok-pratice

# preprocess_sehun 사용예시
from preprocess_sehun import build_rule_dict, preprocess_dataframe
import pandas as pd

df = pd.read_csv("train.csv")
#규칙 생성
rules = build_rule_dict(df, threshold=100)
#전처리 적용
df_processed = preprocess_dataframe(df, rules)
df_processed["input"] = df_processed["input_cleaned"]
df_processed = df_processed.drop(columns=["input_cleaned"])
df_processed.to_csv("train_cleaned.csv", index=False)
