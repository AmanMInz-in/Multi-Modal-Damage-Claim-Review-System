import pandas as pd

df = pd.read_csv("evaluation/sample_predictions.csv")

cols = [
    "gt_issue_type",
    "pred_issue_type",
    "gt_object_part",
    "pred_object_part",
    "gt_claim_status",
    "pred_claim_status",
    "gt_severity",
    "pred_severity"
]

print(df[cols].head())