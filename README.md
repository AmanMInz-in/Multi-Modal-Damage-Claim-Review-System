# Multi-Modal Damage Claim Review System

## Overview

This project implements an AI-powered damage claim verification system that evaluates insurance-style damage claims using multiple information sources:

* Submitted images (primary source of truth)
* User claim conversation
* User claim history
* Evidence requirement rules

The system determines whether image evidence supports, contradicts, or does not provide enough information for the submitted claim.

Supported claim objects:

* Car
* Laptop
* Package

The final output is generated in the required `output.csv` format for challenge submission.

---

# Approach Overview

The solution follows a multimodal evidence review pipeline.

## Step 1: Claim Understanding

The conversation transcript is analyzed to determine:

* Claimed damage type
* Relevant object
* Affected object part

This establishes what evidence should be verified in the images.

---

## Step 2: Image Evidence Analysis

Submitted images are analyzed using Gemini 2.5 Flash Vision.

The model identifies:

* Visible damage
* Issue type
* Object part
* Damage severity
* Image quality issues

Images are treated as the primary source of truth.

---

## Step 3: Evidence Validation

Detected damage is compared against the minimum evidence requirements defined in:

```text
dataset/evidence_requirements.csv
```

The system determines whether sufficient visual evidence exists to evaluate the claim.

---

## Step 4: User History Risk Assessment

Historical user information is retrieved from:

```text
dataset/user_history.csv
```

This information is used only for risk assessment and review prioritization.

Possible risk indicators:

* user_history_risk
* manual_review_required

User history never overrides clear visual evidence.

---

## Step 5: Decision Generation

The system combines:

* Claim conversation
* Visual evidence
* Evidence requirements
* User history

to generate one of the required outcomes:

* supported
* contradicted
* not_enough_information

---

## Step 6: Output Generation

Results are exported to:

```text
output.csv
```

using the exact schema required by the challenge.

---

# System Architecture

```text
claims.csv
     │
     ▼
Claim Extraction
     │
     ▼
Image Loading
     │
     ▼
Gemini Vision Analysis
     │
     ▼
Evidence Validation
     │
     ├── user_history.csv
     │
     └── evidence_requirements.csv
     │
     ▼
Decision Engine
     │
     ▼
output.csv
```

---

# Features

### Claim Understanding

* Extracts damage claim from conversation
* Determines claimed issue and affected part

### Image Analysis

* Reviews one or more submitted images
* Detects visible damage
* Identifies issue type
* Identifies object part
* Estimates severity

### Evidence Validation

* Checks minimum evidence requirements
* Determines whether submitted evidence is sufficient

### Risk Assessment

Flags potential review risks including:

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

### Reliability Features

* Automatic retry logic
* Exponential backoff
* Autosave after every processed claim
* Resume support after interruption

---

# Project Structure

```text
project/

├── generate_output.py
├── evaluate_sample.py
├── requirements.txt
├── README.md
├── .env
│
├── evaluation/
│   ├── sample_predictions.csv
│   └── evaluation_report.md
│
└── dataset/
    ├── claims.csv
    ├── sample_claims.csv
    ├── user_history.csv
    ├── evidence_requirements.csv
    └── images/
```

---

# Dataset Description

## claims.csv

Contains:

* user_id
* image_paths
* user_claim
* claim_object

---

## sample_claims.csv

Labeled examples used for evaluation.

---

## user_history.csv

Contains:

* past claim counts
* review statistics
* history flags
* risk summaries

---

## evidence_requirements.csv

Defines minimum evidence requirements for each:

* object type
* issue family

---

# Installation

## Clone Repository

```bash
git clone <repository_url>
cd project
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas pillow python-dotenv google-generativeai
```

---

# Environment Setup

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

# Running Evaluation

Evaluate against labeled sample claims:

```bash
python evaluate_sample.py
```

or

```bash
python evaluate_sample.py --limit 20
```

Generated files:

```text
evaluation/sample_predictions.csv
evaluation/evaluation_report.md
```

---

# Generating Final Predictions

Run:

```bash
python generate_output.py
```

Generated file:

```text
output.csv
```

---

# Resume and Autosave Strategy

The system automatically saves progress after every processed claim.

Example:

```text
20 claims completed
API quota exhausted
```

After restarting:

```text
Resuming from row 20
```

Processing continues from the next unfinished claim.

This prevents loss of progress due to:

* API limits
* Network interruptions
* Unexpected crashes

---

# Evaluation Methodology

The system was evaluated using:

```text
dataset/sample_claims.csv
```

Performance was measured against ground-truth labels for:

* issue_type
* object_part
* claim_status
* severity

Evaluation outputs are stored in:

```text
evaluation/
```

---

# Operational Analysis

## Approximate Model Calls

Sample Evaluation:

* ~20 Gemini API calls

Full Test Dataset:

* ~44 Gemini API calls

---

## Approximate Images Processed

Sample Dataset:

* ~40 images

Full Dataset:

* 100+ images

---

## Latency

Average claim processing time:

* 5–15 seconds

Estimated full dataset runtime:

* 5–15 minutes

depending on image count and network conditions.

---

## Rate Limit Handling

Implemented:

* Retry mechanism
* Exponential backoff
* Resume support
* Autosave support

These reduce failures caused by API quota limits.

---

## Cost Considerations

Development was performed using Gemini 2.5 Flash.

Estimated usage:

* One vision model call per claim
* Low operational cost for small datasets

---

# Technologies Used

* Python
* Pandas
* Pillow
* Python Dotenv
* Gemini 2.5 Flash
* CSV-based evaluation pipeline

---

# Submission Files

The final submission includes:

```text
code.zip
output.csv
chat_transcript
```

---

# Author

Developed for the Multi-Modal Evidence Review Challenge.

Focus Areas:

* Multimodal reasoning
* Image-based verification
* Evidence validation
* Risk-aware claim assessment
* Robust batch processing
