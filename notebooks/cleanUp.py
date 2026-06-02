import pandas as pd

INPUT_CSV  = "data/ielts_relabeled.csv"
OUTPUT_CSV = "data/ielts_relabeled.csv"

df = pd.read_csv(INPUT_CSV)
print(f"Before: {df.shape}")

# Drop truncated duplicate column
if "Examiner_Commen" in df.columns:
    df = df.drop(columns=["Examiner_Commen"])
    print("Dropped column: Examiner_Commen")

# Drop rows with missing labels
before = len(df)
df = df.dropna(subset=["Task_Response", "Coherence_Cohesion",
                        "Lexical_Resource", "Range_Accuracy", "Examiner_Comment"])
dropped = before - len(df)
print(f"Dropped {dropped} row(s) with missing labels")

df.to_csv(OUTPUT_CSV, index=False)
print(f"After: {df.shape}")
print(f"Saved to {OUTPUT_CSV}")