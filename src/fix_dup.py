import pandas as pd

df = pd.read_csv("data/ielts_labeled.csv")
print(f"Before: {df.shape}")

old = df[df["Task_Response"].isna()].copy()
new = df[df["Task_Response"].notna()].copy()

print(f"Old rows (comments): {len(old)}")
print(f"New rows (scores):   {len(new)}")

old = old.reset_index(drop=True)
new = new.reset_index(drop=True)

# the ONE real operation — bring comments from old into new
new["Examiner_Commen"] = old["Examiner_Commen"]

# handle duplicate column
if "Grammar_Range_Accuracy" in new.columns:
    new["Range_Accuracy"] = new["Range_Accuracy"].fillna(new["Grammar_Range_Accuracy"])
    new = new.drop(columns=["Grammar_Range_Accuracy"])

print(f"\nFinal shape: {new.shape}")
print(f"Missing values:\n{new.isnull().sum()}")

new.to_csv("data/ielts_labeled.csv", index=False)
print("Saved!")