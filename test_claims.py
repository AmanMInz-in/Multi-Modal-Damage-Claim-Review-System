import pandas as pd

claims = pd.read_csv("dataset/claims.csv")

print("Total Claims:", len(claims))
print()

print(claims.head())