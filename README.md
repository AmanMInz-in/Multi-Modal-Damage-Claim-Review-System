# Multi-Modal Damage Claim Review System

## Overview

This project is an AI-powered damage claim verification system that analyzes images, claim conversations, user history, and evidence requirements to determine whether a submitted damage claim is supported, contradicted, or lacks sufficient information.

The system supports three claim object categories:

* Car
* Laptop
* Package

Image evidence is treated as the primary source of truth. User conversations provide claim context, while user history contributes risk indicators without overriding visual evidence.

---

## Features

### Claim Understanding

* Extracts the actual damage claim from conversation transcripts.
* Identifies the claimed object and affected component.

### Image Analysis

* Reviews one or more submitted images.
* Detects visible damage.
* Identifies issue type.
* Identifies object part.
* Estimates severity.

### Evidence Validation

* Checks minimum evidence requirements.
* Determines whether submitted evidence is sufficient for evaluation.

### Decision Engine

Produces one of:

* supported
* contradicted
* not_enough_information

### Risk Assessment

Flags potential review risks:

* blurry_image
* cropped_or_obstructed
* low_light_or_glare
* wrong_angle
* wrong_object
* wrong_object_part
* damage_not_visible
* claim_mismatch
* possible_manipulation
* non_original_image
* text_instruction_present
* user_history_risk
* manual_review_required

### Resume and Autosave Support

The system automatically:

* Saves progress after every processed claim.
* Resumes from the last completed row if interrupted.
* Prevents loss of progress due to API limits or failures.

---

## Dataset Structure

dataset/

├── claims.csv

├── sample_claims.csv

├── user_history.csv

├── evidence_requirements.csv

└── images/

```
├── sample/

└── test/
```

---

## Output Schema

The generated output.csv contains:

| Column                       |
| ---------------------------- |
| user_id                      |
| image_paths                  |
| user_claim                   |
| claim_object                 |
| evidence_standard_met        |
| evidence_standard_met_reason |
| risk_flags                   |
| issue_type                   |
| object_part                  |
| claim_status                 |
| claim_status_justification   |
| supporting_image_ids         |
| valid_image                  |
| severity                     |

---

## Model Used

Gemini 2.5 Flash

The model is used for:

* Visual damage analysis
* Object identification
* Damage classification
* Severity estimation
* Evidence reasoning

---

## Installation

### Clone Project

```bash
git clone <repository_url>
cd project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```bash
pip install pandas pillow python-dotenv google-generativeai
```

---

## Environment Variables

Create a .env file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## Running Evaluation

Evaluate using provided labeled samples:

```bash
python evaluate_sample.py
```

Or:

```bash
python evaluate_sample.py --limit 20
```

Outputs:

* evaluation/sample_predictions.csv
* evaluation/evaluation_report.md

---

## Generating Predictions

Run:

```bash
python generate_output.py
```

Output:

```text
output.csv
```

---

## Resume Logic

The system automatically resumes from the last processed claim.

Example:

```text
output.csv contains 20 rows

Next execution starts from row 21
```

This prevents repeated processing when API limits are reached.

---

## Autosave Strategy

After each processed claim:

1. Result is appended.
2. output.csv is updated.
3. Progress is preserved immediately.

This minimizes loss from:

* API quota limits
* Network failures
* Unexpected crashes

---

## Evaluation Methodology

The system was evaluated using sample_claims.csv.

Metrics compared:

* issue_type
* object_part
* claim_status
* severity

Outputs were compared against provided ground-truth labels.

---

## Operational Analysis

### Approximate Model Calls

Sample Evaluation:

* 20 claims
* ~20 Gemini API calls

Full Dataset:

* 44 claims
* ~44 Gemini API calls

### Approximate Images Processed

* Sample set: ~40 images
* Test set: ~100+ images

### Estimated Latency

Average per claim:

* 5–15 seconds

Full dataset:

* Approximately 5–15 minutes

depending on network conditions and API rate limits.

### Rate Limit Handling

Implemented:

* Retry mechanism
* Exponential backoff
* Resume support
* Autosave support

### Cost Considerations

Gemini Free Tier was used during development.

Production deployment may require:

* Paid Gemini API usage
* Higher quota limits

---

## Project Workflow

1. Load claim data.
2. Load user history.
3. Load evidence requirements.
4. Load images.
5. Analyze images with Gemini.
6. Extract visible damage information.
7. Apply evidence validation.
8. Generate claim decision.
9. Save results.
10. Export output.csv.

---

## Submission Files

Submission package contains:

```text
code.zip
output.csv
chat_transcript
```

---

## Author

Built as a solution for the Multi-Modal Evidence Review Challenge.

Technologies:

* Python
* Pandas
* Pillow
* Gemini 2.5 Flash
* CSV-based evaluation pipeline
