# src/reset_scores.py
import pandas as pd

df = pd.read_csv("data/ielts_labeled.csv")

df["Task_Response"] = None
df["Coherence_Cohesion"] = None
df["Lexical_Resource"] = None
df["Range_Accuracy"] = None

df.to_csv("data/ielts_labeled.csv", index=False)
print("Scores reset! Comments preserved.")
print("Missing scores:", df["Task_Response"].isnull().sum())
print("Missing comments:", df["Examiner_Commen"].isnull().sum())