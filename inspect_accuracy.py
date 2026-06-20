import pandas as pd

df = pd.read_csv("evaluation/sample_predictions.csv")

print(df[[
    "gt_issue_type",
    "pred_issue_type",
    "gt_severity",
    "pred_severity"
]])