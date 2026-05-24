import pandas as pd

df = pd.read_csv("data/ielts_labeled.csv")

print("Before:", df.columns.tolist())

# Merge all three comment columns into one
df["Examiner_Commen"] = (
    df["Examiner_Commen"]
    .fillna(df["Examiner_Comment"])
    .fillna(df["Examiner_Comments"])
)

# Drop duplicate columns
df = df.drop(columns=["Examiner_Comment", "Examiner_Comments"])

print("After:", df.columns.tolist())
print("Missing comments:", df["Examiner_Commen"].isnull().sum())

df.to_csv("data/ielts_labeled.csv", index=False)
print("Saved!")