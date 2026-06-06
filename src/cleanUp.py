import pandas as pd

df = pd.read_csv("data/ielts_relabeled_v3.csv")

# drop rows with missing labels
df = df.dropna(subset=["Task_Response", "Coherence_Cohesion", "Lexical_Resource", "Range_Accuracy"])

# cap invalid scores
for col in ["Task_Response", "Coherence_Cohesion", "Lexical_Resource", "Range_Accuracy"]:
    df[col] = df[col].clip(1.0, 9.0)

df.to_csv("data/ielts_relabeled_v3.csv", index=False)
print(f"Saved {len(df)} rows")
print(df[["Task_Response","Coherence_Cohesion","Lexical_Resource","Range_Accuracy"]].describe())