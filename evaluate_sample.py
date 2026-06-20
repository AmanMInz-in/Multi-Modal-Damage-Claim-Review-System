"""
evaluate_sample.py

Evaluates Gemini 2.5 Flash Vision on an insurance damage-claim dataset.

For every row in dataset/sample_claims.csv this script:
  1. Loads the claim's images
  2. Sends them + claim text to Gemini 2.5 Flash for structured prediction
  3. Parses the JSON response (with retries + safeguards)
  4. Compares predicted fields against ground-truth labels
  5. Writes per-row predictions to evaluation/sample_predictions.csv
  6. Prints an accuracy summary report

Usage:
    python evaluate_sample.py
    python evaluate_sample.py --limit 20          # quick smoke test
    python evaluate_sample.py --model gemini-2.5-flash

Requires a .env file (or exported env var) with:
    GOOGLE_API_KEY=your_api_key_here
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:
    sys.exit(
        "Missing dependency 'google-generativeai'. Install with:\n"
        "    pip install google-generativeai pandas pillow python-dotenv"
    )

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DATASET_DIR = Path("dataset")
CLAIMS_FILE = DATASET_DIR / "sample_claims.csv"
OUTPUT_DIR = Path("evaluation")
OUTPUT_FILE = OUTPUT_DIR / "sample_predictions.csv"

PREDICTED_FIELDS = ["issue_type", "object_part", "claim_status", "severity"]

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
REQUEST_PAUSE_SECONDS = 0.5  # gentle pacing between calls

PROMPT_TEMPLATE = PROMPT_TEMPLATE = """
You are an insurance damage claim reviewer.

Claim object: {claim_object}

User claim:
{claim_text}

Analyze ALL submitted images together.

Use the images as the primary source of truth.

IMPORTANT:

Return ONLY valid JSON.

Return EXACTLY ONE value for issue_type.
Return EXACTLY ONE value for object_part.
Return EXACTLY ONE value for claim_status.
Return EXACTLY ONE value for severity.

Allowed claim_status values:

supported
contradicted
not_enough_information

Allowed severity values:

none
low
medium
high
unknown

Allowed issue_type values:

dent
scratch
crack
glass_shatter
broken_part
missing_part
torn_packaging
crushed_packaging
water_damage
stain
none
unknown

Car object_part values:

front_bumper
rear_bumper
door
hood
windshield
side_mirror
headlight
taillight
fender
quarter_panel
body
unknown

Laptop object_part values:

screen
keyboard
trackpad
hinge
lid
corner
port
base
body
unknown

Package object_part values:

box
package_corner
package_side
seal
label
contents
item
unknown

DO NOT USE:
shattered
cracked
damaged
severe
minor
moderate

Map:
shattered -> glass_shatter
cracked -> crack
severe -> high
moderate -> medium
minor -> low

Return ONLY this JSON schema:

{{
  "issue_type":"",
  "object_part":"",
  "claim_status":"",
  "severity":""
}}
"""
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_images(image_paths_field, base_dir: Path):
    """
    Parses the image_paths column (supports comma/semicolon/pipe separated
    strings or a JSON list) and loads each as a PIL.Image.

    Returns (images, missing_paths, load_errors)
    """
    images = []
    missing = []
    errors = []

    if pd.isna(image_paths_field) or not str(image_paths_field).strip():
        return images, missing, errors

    raw = str(image_paths_field).strip()

    # Try JSON list first (e.g. '["a.jpg", "b.jpg"]')
    paths = None
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                paths = [str(p).strip() for p in parsed]
        except json.JSONDecodeError:
            paths = None

    if paths is None:
        # Fall back to delimiter-separated string
        for delim in [",", ";", "|"]:
            if delim in raw:
                paths = [p.strip() for p in raw.split(delim) if p.strip()]
                break
        if paths is None:
            paths = [raw]

    for p in paths:
        candidate = Path(p)
        if not candidate.is_absolute():
            # Try relative to base_dir, then relative to cwd
            alt = base_dir / candidate
            candidate = alt if alt.exists() else candidate

        if not candidate.exists():
            missing.append(str(p))
            continue

        try:
            img = Image.open(candidate)
            img.load()  # force read now so we catch corrupt files here
            images.append(img.convert("RGB"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{p}: {e}")

    return images, missing, errors


def extract_json(text: str):
    """
    Pulls the first {...} block out of a model response and parses it.
    Handles cases where the model wraps JSON in markdown fences or adds
    stray preamble/postamble text.
    """
    if not text:
        return None, "empty response"

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Direct parse attempt
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass

    # Fallback: regex out the largest {...} block
    match = JSON_BLOCK_RE.search(cleaned)
    if not match:
        return None, "no JSON object found in response"

    candidate = match.group(0)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"


def normalize(value):
    """Normalize a value for fair string comparison (case/whitespace)."""
    if value is None:
        return ""
    return str(value).strip().lower()


def call_gemini_with_retries(model, prompt, images, max_retries=MAX_RETRIES):
    """
    Calls Gemini with retry + backoff on transient errors, and returns
    (parsed_json_dict_or_None, raw_text, error_message_or_None).
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            content = [prompt] + images
            response = model.generate_content(content)
            raw_text = (response.text or "").strip() if hasattr(response, "text") else ""
            parsed, parse_err = extract_json(raw_text)
            if parsed is not None:
                return parsed, raw_text, None
            last_error = parse_err or "unparseable response"
        except Exception as e:  # noqa: BLE001
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return None, "", last_error


# --------------------------------------------------------------------------- #
# Main evaluation logic
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini Vision on insurance claims dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--claims-file", type=str, default=str(CLAIMS_FILE), help="Path to sample_claims.csv"
    )
    parser.add_argument(
        "--image-base-dir", type=str, default=str(DATASET_DIR),
        help="Base directory used to resolve relative image paths."
    )
    args = parser.parse_args()

    # --- Setup ---
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "No API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your "
            "environment or a .env file."
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    claims_path = Path(args.claims_file)
    if not claims_path.exists():
        sys.exit(f"Claims file not found: {claims_path}")

    df = pd.read_csv(claims_path)

    required_cols = {"user_id", "image_paths", "user_claim", "claim_object"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        sys.exit(f"sample_claims.csv is missing required columns: {missing_cols}")

    for col in PREDICTED_FIELDS:
        if col not in df.columns:
            print(f"Warning: ground-truth column '{col}' not found; that accuracy will be skipped.")

    if args.limit:
        df = df.head(args.limit).copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_base_dir = Path(args.image_base_dir)

    results = []
    total = len(df)

    print(f"Evaluating {total} claims with model '{args.model}'...\n")

    for idx, row in df.iterrows():
        row_num = idx + 1
        user_id = row.get("user_id", "")
        claim_object = row.get("claim_object", "")
        user_claim = row.get("user_claim", "")

        print(f"[{row_num}/{total}] user_id={user_id} claim_object={claim_object!r}", end=" ... ")

        images, missing_paths, load_errors = load_images(row.get("image_paths"), image_base_dir)

        record = {
            "row_index": idx,
            "user_id": user_id,
            "claim_object": claim_object,
            "user_claim": user_claim,
            "num_images_loaded": len(images),
            "missing_image_paths": "; ".join(missing_paths) if missing_paths else "",
            "image_load_errors": "; ".join(load_errors) if load_errors else "",
        }

        # Ground truth passthrough
        for col in PREDICTED_FIELDS:
            record[f"gt_{col}"] = row.get(col, "")

        if not images:
            print("SKIPPED (no images could be loaded)")
            record["status"] = "skipped_no_images"
            record["raw_response"] = ""
            record["api_error"] = "no images available"
            for col in PREDICTED_FIELDS:
                record[f"pred_{col}"] = ""
                record[f"correct_{col}"] = False
            results.append(record)
            continue

        prompt = PROMPT_TEMPLATE.format(claim_object=claim_object, claim_text=user_claim)
        parsed, raw_text, error = call_gemini_with_retries(model, prompt, images)

        record["raw_response"] = raw_text
        record["api_error"] = error or ""

        if parsed is None:
            print(f"FAILED ({error})")
            record["status"] = "failed"
            for col in PREDICTED_FIELDS:
                record[f"pred_{col}"] = ""
                record[f"correct_{col}"] = False
        else:
            record["status"] = "ok"
            for col in PREDICTED_FIELDS:
                pred_value = parsed.get(col, "")
                record[f"pred_{col}"] = pred_value
                gt_value = row.get(col, "")
                record[f"correct_{col}"] = (
                    normalize(pred_value) == normalize(gt_value) and str(gt_value).strip() != ""
                )
            print("OK")

        results.append(record)
        time.sleep(REQUEST_PAUSE_SECONDS)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPredictions saved to {OUTPUT_FILE}\n")

    print_summary(results_df, total)


def print_summary(results_df: pd.DataFrame, total: int):
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    ok_count = (results_df["status"] == "ok").sum()
    failed_count = (results_df["status"] == "failed").sum()
    skipped_count = (results_df["status"] == "skipped_no_images").sum()

    print(f"Total claims:        {total}")
    print(f"Successful calls:    {ok_count}")
    print(f"Failed calls:        {failed_count}")
    print(f"Skipped (no images): {skipped_count}")
    print("-" * 60)

    scored_df = results_df[results_df["status"] == "ok"]
    n_scored = len(scored_df)

    if n_scored == 0:
        print("No successfully evaluated rows; cannot compute accuracy.")
        print("=" * 60)
        return

    print(f"Accuracy (computed over {n_scored} successfully-evaluated rows "
          f"with non-empty ground truth):\n")

    overall_correct = 0
    overall_total = 0

    for field in PREDICTED_FIELDS:
        gt_col = f"gt_{field}"
        correct_col = f"correct_{field}"
        if gt_col not in scored_df.columns:
            continue

        evaluable = scored_df[scored_df[gt_col].astype(str).str.strip() != ""]
        n_eval = len(evaluable)
        if n_eval == 0:
            print(f"  {field:15s}: no ground truth available")
            continue

        n_correct = evaluable[correct_col].sum()
        acc = n_correct / n_eval
        overall_correct += n_correct
        overall_total += n_eval
        print(f"  {field:15s}: {acc:.2%}  ({n_correct}/{n_eval})")

    print("-" * 60)
    if overall_total > 0:
        print(f"  {'OVERALL':15s}: {overall_correct / overall_total:.2%}  "
              f"({overall_correct}/{overall_total} field predictions)")
    print("=" * 60)


if __name__ == "__main__":
    main()
