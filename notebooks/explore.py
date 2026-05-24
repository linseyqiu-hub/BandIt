import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd

# load
df = pd.read_csv('../data/ielts_writing_dataset.csv')

# basic info
print('=== Shape ===')
print(df.shape)

print('\n=== Columns ===')
print(df.columns.tolist())

print('\n=== First row ===')
print(df.iloc[0])

print('\n=== Data types ===')
print(df.dtypes)

print('\n=== Missing values ===')
print(df.isnull().sum())

print('\n=== Score distribution (Overall) ===')
print(df['Overall'].value_counts().sort_index())

print('\n=== Task type distribution ===')
print(df['Task_Type'].value_counts())

print('\n=== Essay length stats ===')
df['essay_length'] = df['Essay'].apply(lambda x: len(str(x).split()))
print(df['essay_length'].describe())
# our results:
# === Shape ===
# (1435, 9)

# === Columns ===
# ['Task_Type', 'Question', 'Essay', 'Examiner_Commen', 'Task_Response', 'Coherence_Cohesion', 'Lexical_Resource', 'Range_Accuracy', 'Overall']

# === First row ===
# Task_Type                                                             1
# Question              The bar chart below describes some changes abo...
# Essay                 Between 1995 and 2010, a study was conducted r...
# Examiner_Commen                                                     NaN
# Task_Response                                                       NaN
# Coherence_Cohesion                                                  NaN
# Lexical_Resource                                                    NaN
# Range_Accuracy                                                      NaN
# Overall                                                             5.5
# Name: 0, dtype: object

# === Data types ===
# Task_Type               int64
# Question               object
# Essay                  object
# Examiner_Commen        object
# Task_Response         float64
# Coherence_Cohesion    float64
# Lexical_Resource      float64
# Range_Accuracy        float64
# Overall               float64
# dtype: object

# === Missing values ===
# Task_Type                0
# Question                 0
# Essay                    0
# Examiner_Commen       1373
# Task_Response         1435
# Coherence_Cohesion    1435
# Lexical_Resource      1435
# Range_Accuracy        1435
# Overall                  0
# dtype: int64

# === Score distribution (Overall) ===
# Overall
# 1.0      1
# 3.0      2
# 3.5      5
# 4.0     11
# 4.5     21
# 5.0    104
# 5.5    176
# 6.0    264
# 6.5    250
# 7.0    254
# 7.5    138
# 8.0    137
# 8.5     35
# 9.0     37
# Name: count, dtype: int64

# === Task type distribution ===
# Task_Type
# 2    793
# 1    642
# Name: count, dtype: int64

# === Essay length stats ===
# count    1435.000000
# mean      256.732404
# std        81.324270
# min       116.000000
# 25%       181.000000
# 50%       259.000000
# 75%       307.000000
# max       577.000000
# Name: essay_length, dtype: float64