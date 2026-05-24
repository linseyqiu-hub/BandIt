import os
import tiktoken
import pandas as pd

df = pd.read_csv("data/ielts_writing_dataset.csv")

enc = tiktoken.get_encoding("cl100k_base")
count = 0
for i, row in df.iterrows():
    prompt = f"Question: {row['Question']}\n\nEssay:\n{row['Essay']}"
    tokens = len(enc.encode(prompt))
    if tokens > 512:
        count += 1
print(count, "rows exceed 512 tokens")