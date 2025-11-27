import torch
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset


#######################################################################
# 0) 전처리
os.system("pip install triton transformers==4.41.2 accelerate==0.28.0 bitsandbytes peft==0.10.0")
import os
#######################################################################
print("=== 📌 STEP 1: Train 전처리 시작 ===")

from preprocess import (
    build_rules_from_train,
    restore_with_jamo_rules
)

train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")  

# 전처리 규칙 생성
rules, stats = build_rules_from_train(
    train_raw,
    input_col="input",
    output_col="output",
    min_total=50,
    min_top_prob=0.8
)

# train input 복원
train_raw["input"] = train_raw["input"].apply(lambda x: restore_with_jamo_rules(x, rules))
train_raw.to_csv("train_cleaned.csv", index=False)

print("✔ train_cleaned.csv 저장 완료")


#######################################################################
# 1) Dict Baseline
#######################################################################
print("\n=== 📌 STEP 2: Dict Baseline ===")

match_dict = {}
for inp, out in zip(train_raw["input"], train_raw["output"]):
    for iw, ow in zip(inp.split(), out.split()):
        match_dict[iw] = ow

def replace_words(x):
    return " ".join([match_dict.get(w, w) for w in x.split()])

converted_reviews = test_raw["input"].apply(replace_words).tolist()

submission_base = pd.read_csv("sample_submission.csv")
submission_base["output"] = converted_reviews
submission_base.to_csv("baseline_submission_dict.csv", index=False)

print("✔ baseline_submission_dict.csv 저장 완료")


#######################################################################
# 2) LLM 준비 (gemma-ko-2b)
#######################################################################
print("\n=== 📌 STEP 3: LLM 로드 ===")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", device)

# 4비트 로딩
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "beomi/gemma-ko-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)


#######################################################################
# 3) LoRA 세팅
#######################################################################
print("\n=== 📌 STEP 4: LoRA 준비 ===")

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_disable()

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)


#######################################################################
# 4) Dataset 준비 (instruction tuning)
#######################################################################
dataset = load_dataset("csv", data_files="train_cleaned.csv")["train"]
split = dataset.train_test_split(test_size=0.2, seed=42)
def formatting(example):
    prompt = (
        "<|user|>\n"
        "너는 난독화된 한국어 리뷰를 자연스럽고 원래 의미대로 복원하는 모델이야.\n"
        "아래의 input 문장을 자연스러운 한국어 문장으로 바꿔줘.\n\n"
        "예시 1:\n"
        "input: 덥몁 닝쿨 학꾼데욬 ㅋ\n"
        "output: 덥지 않고 시원하네요 ㅋ\n\n"
        "예시 2:\n"
        "input: 잚많 작꼬 갉 태 좋네욥.\n"
        "output: 잠만 자고 갈 때 좋네요.\n\n"
        f"input: {example['input']}\n"
        "<|assistant|>\n"
        f"{example['output']}"
    )

    tokenized = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


     
train_set = split["train"].map(formatting, remove_columns=split["train"].column_names)
eval_set  = split["test"].map(formatting, remove_columns=split["test"].column_names)


#######################################################################
# 5) Training
#######################################################################
print("\n=== 📌 STEP 5: LoRA 학습 시작 ===")

training_args = TrainingArguments(
    output_dir="lora_out",
    per_device_train_batch_size=2,
    max_steps=1500,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="no",
    evaluation_strategy="no",
    report_to="none"
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    data_collator=collator,
)

model.config.use_cache = False
trainer.train()

print("✔ LoRA 학습 완료")


#######################################################################
# 6) Inference
#######################################################################
print("\n=== 📌 STEP 6: LLM Inference 시작 ===")

merged = model.merge_and_unload()
merged.eval()

restored = []

for _, row in tqdm(test_raw.iterrows(), desc="LLM generating"):
    query = row["input"]

    prompt = (
    "<|user|>\n"
    "너는 난독화된 한국어 리뷰를 자연스럽고 원래 의미대로 복원하는 모델이야.\n"
    f"input: {query}\n"
    "<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        out = merged.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id
        )

    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = gen.split("<|assistant|>")[-1].strip()
    restored.append(answer)


#######################################################################
# 7) 제출 저장
#######################################################################
submission_llm = pd.read_csv("sample_submission.csv")
submission_llm["output"] = restored
submission_llm.to_csv("baseline_submission_llm.csv", index=False)

print("✔ baseline_submission_llm.csv 저장 완료")