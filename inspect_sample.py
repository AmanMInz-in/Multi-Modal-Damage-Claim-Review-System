import pandas as pd

sample = pd.read_csv("dataset/sample_claims.csv")

row = sample.iloc[0]

for col in sample.columns:
    print(f"{col}:")
    print(row[col])
    print("-" * 50)