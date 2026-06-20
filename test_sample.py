import pandas as pd

sample = pd.read_csv("dataset/sample_claims.csv")

print(sample.columns.tolist())
print()
print(sample.head(3))