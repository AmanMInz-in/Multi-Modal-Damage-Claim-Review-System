"""
generate_output.py

Runs the same Gemini-based claim analysis used in evaluate_sample.py, but over
the FULL dataset (dataset/claims.csv, 44 claims) instead of the sample file,
and writes a complete output.csv with all required columns.

-----------------------------------------------------------------------------
IMPORTANT — ADAPT THIS TO YOUR evaluate_sample.py
-----------------------------------------------------------------------------
I did not have access to your actual evaluate_sample.py in this session, so
the pieces below marked with `# ADAPT:` are my best-guess reconstruction of
"the same Gemini analysis logic" (SDK = google-generativeai, model =
gemini-2.5-flash, JSON-mode prompting). If your real script differs (e.g. you
use the new `google-genai` Client SDK, a different prompt, or a different
response schema), just replace the body of `call_gemini_for_claim()` with
your existing logic — everything else (CSV I/O, image loading, retries,
deterministic fallback rules, output columns) will keep working unchanged.
-----------------------------------------------------------------------------

Output columns produced for every row:
    user_id, image_paths, user_claim, claim_object,
    evidence_standard_met, evidence_standard_met_reason,
    risk_flags, issue_type, object_part, claim_status,
    claim_status_justification, supporting_image_ids, valid_image, severity
"""

import os
import csv
import json
import time
import random
import logging
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
import pandas as pd

# ADAPT: swap this import / client setup for whatever evaluate_sample.py uses
import google.generativeai as genai

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

load_dotenv()  # reuse the same .env-based setup as evaluate_sample.py

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file."
    )

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"  # ADAPT: match evaluate_sample.py if different

INPUT_CSV = Path("dataset/claims.csv")
OUTPUT_CSV = Path("output0.csv")

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2.0  # exponential backoff: base * 2^attempt + jitter

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# Fields Gemini is expected to return directly (per requirement #9, only
# these 4 are assumed to reliably come back from the model; everything else
# is filled in deterministically if missing).
GEMINI_DIRECT_FIELDS = ["issue_type", "object_part", "claim_status", "severity"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_output")

model = genai.GenerativeModel(MODEL_NAME)


# -----------------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------------

def parse_image_paths(raw_value):
    """
    Splits the image_paths cell into a list of individual paths.

    ADAPT: if your claims.csv stores paths differently (semicolon, pipe,
    JSON list, etc.), change the split logic here.
    """
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    raw_value = str(raw_value).strip()
    if not raw_value:
        return []
    # Try JSON list first (e.g. '["a.jpg", "b.jpg"]')
    if raw_value.startswith("["):
        try:
            parsed = json.loads(raw_value)
            return [str(p).strip() for p in parsed if str(p).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    # Fall back to comma/semicolon/pipe separated string
    for sep in [",", ";", "|"]:
        if sep in raw_value:
            return [p.strip() for p in raw_value.split(sep) if p.strip()]
    return [raw_value]


def load_images(image_paths):
    """
    Loads each image path with PIL.

    Returns a list of dicts: {"path": str, "image": PIL.Image|None, "valid": bool}
    """
    loaded = []
    for path in image_paths:
        entry = {"path": path, "image": None, "valid": False}
        try:
            full_path = Path("dataset") / path
            with Image.open(full_path) as img:
                img.load()  # force decode now, catches truncated/corrupt files
                entry["image"] = img.copy()
                entry["valid"] = True
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            logger.warning("Could not load image '%s': %s", path, e)
        loaded.append(entry)
    return loaded


# -----------------------------------------------------------------------------
# Gemini call with retry handling
# -----------------------------------------------------------------------------

def build_prompt(user_claim, claim_object):
    """
    ADAPT: replace with the exact prompt text used in evaluate_sample.py.
    This is a reasonable reconstruction based on the required output schema.
    """
    return f"""You are evaluating a product/object claim against the provided images.

Claim object: {claim_object}
User claim: "{user_claim}"

Analyze the images and respond with ONLY a JSON object (no markdown fences,
no extra text) with exactly these fields:

{{
  "issue_type": "<short category of the issue found, or 'none'>",
  "object_part": "<the specific part of the object the claim/issue relates to>",
  "claim_status": "<one of: supported, contradicted, not_enough_information>",
  "severity": "<one of: none, low, medium, high, unknown>"
}}
"""


def call_gemini_for_claim(user_claim, claim_object, images):
    """
    ADAPT: replace the body of this function with the exact Gemini call from
    evaluate_sample.py (same model, same prompt, same content packing) if it
    differs from this reconstruction.

    Returns a dict with at least: issue_type, object_part, claim_status, severity
    (any subset may be missing if Gemini's response was incomplete).
    """
    prompt = build_prompt(user_claim, claim_object)

    content = [prompt]
    for entry in images:
        if entry["valid"] and entry["image"] is not None:
            content.append(entry["image"])

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(content)
            return parse_gemini_json(response.text)
        except Exception as e:  # noqa: BLE001 - broad on purpose for API/network errors
            last_exception = e
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)

    logger.error("Gemini call failed after %d attempts: %s", MAX_RETRIES, last_exception)
    return {}


def parse_gemini_json(raw_text):
    """
    Robustly parses Gemini's JSON response, stripping markdown code fences
    if present and falling back gracefully on parse errors.
    """
    if not raw_text:
        return {}
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse Gemini response as JSON: %r", raw_text[:300])
        return {}


# -----------------------------------------------------------------------------
# Deterministic fallback rules for derived fields
# -----------------------------------------------------------------------------

def compute_deterministic_fields(gemini_result, image_entries):
    """
    Fills in evidence_standard_met, evidence_standard_met_reason, risk_flags,
    claim_status_justification, supporting_image_ids, and valid_image whenever
    Gemini did not supply them directly (per requirement #9, Gemini is only
    expected to reliably return issue_type/object_part/claim_status/severity).

    Rules are deterministic and explainable:

    - valid_image: True only if at least one image loaded successfully.
    - supporting_image_ids: indices (0-based) of successfully loaded images
      that were actually sent to Gemini as evidence.
    - evidence_standard_met: True only if there is at least one valid
      supporting image AND claim_status indicates the claim was actually
      assessed (i.e. not "inconclusive" and not missing).
    - evidence_standard_met_reason: short human-readable explanation tied
      directly to the rule above.
    - risk_flags: list built from simple, explainable conditions (no
      supporting images, high/critical severity, unsupported claim, low/no
      evidence).
    - claim_status_justification: short sentence built from claim_status,
      issue_type, object_part, and severity.
    """
    valid_flags = [e["valid"] for e in image_entries]
    supporting_image_ids = [i for i, v in enumerate(valid_flags) if v]
    valid_image = any(valid_flags)

    claim_status = gemini_result.get(
    "claim_status",
    "not_enough_information"
) or "not_enough_information"
    issue_type = gemini_result.get("issue_type", "unknown") or "unknown"
    object_part = gemini_result.get("object_part", "unknown") or "unknown"
    severity = (gemini_result.get("severity", "none") or "none").lower()

    evidence_standard_met = bool(supporting_image_ids) and claim_status not in (
        "not_enough_information",
        "",
    )

    if not supporting_image_ids:
        evidence_standard_met_reason = "No valid images were available to support or refute the claim."
    elif claim_status == "not_enough_information":
        evidence_standard_met_reason = "Images were available but the analysis could not reach a conclusive determination."
    else:
        evidence_standard_met_reason = (
            f"{len(supporting_image_ids)} valid image(s) were available and used to "
            f"reach a '{claim_status}' determination."
        )

    risk_flags = []
    if not valid_image:
        risk_flags.append("no_valid_image")
    if claim_status == "contradicted":
        risk_flags.append("unsupported_claim")
    if claim_status == "not_enough_information":
        risk_flags.append("inconclusive_evidence")
    if severity == "high":
        risk_flags.append(f"{severity}_severity")
    if not evidence_standard_met:
        risk_flags.append("evidence_standard_not_met")

    claim_status_justification = (
        f"Claim status '{claim_status}' based on issue type '{issue_type}' "
        f"affecting '{object_part}' with severity '{severity}'."
    )

    return {
        "evidence_standard_met": evidence_standard_met,
        "evidence_standard_met_reason": evidence_standard_met_reason,
        "risk_flags": risk_flags,
        "claim_status_justification": claim_status_justification,
        "supporting_image_ids": supporting_image_ids,
        "valid_image": valid_image,
    }


# -----------------------------------------------------------------------------
# Row processing
# -----------------------------------------------------------------------------

def process_row(row):
    user_id = row.get("user_id")
    raw_image_paths = row.get("image_paths")
    user_claim = row.get("user_claim", "")
    # ADAPT: if claim_object isn't a column in claims.csv, derive it from
    # user_claim here instead (e.g. via a separate Gemini call or simple
    # extraction logic from evaluate_sample.py).
    claim_object = row.get("claim_object", "")

    image_paths = parse_image_paths(raw_image_paths)
    image_entries = load_images(image_paths)

    gemini_result = call_gemini_for_claim(user_claim, claim_object, image_entries)

    derived = compute_deterministic_fields(gemini_result, image_entries)

    output_row = {
        "user_id": user_id,
        "image_paths": raw_image_paths,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": gemini_result.get(
            "evidence_standard_met", derived["evidence_standard_met"]
        ),
        "evidence_standard_met_reason": gemini_result.get(
            "evidence_standard_met_reason", derived["evidence_standard_met_reason"]
        ),
        "risk_flags": json.dumps(gemini_result.get("risk_flags", derived["risk_flags"])),
        "issue_type": gemini_result.get("issue_type", "unknown"),
        "object_part": gemini_result.get("object_part", "unknown"),
        "claim_status": gemini_result.get("claim_status", "not_enough_information"),
        "claim_status_justification": gemini_result.get(
            "claim_status_justification", derived["claim_status_justification"]
        ),
        "supporting_image_ids": json.dumps(
            gemini_result.get("supporting_image_ids", derived["supporting_image_ids"])
        ),
        "valid_image": gemini_result.get("valid_image", derived["valid_image"]),
        "severity": gemini_result.get("severity", "none"),
    }
    return output_row


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}. Expected 44 claims at this path."
        )

    df = pd.read_csv(INPUT_CSV)

    # Resume support
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        already_done = len(existing)

        logger.info("Resuming from row %d", already_done)

        df = df.iloc[already_done:]
        results = existing.to_dict("records")
    else:
        already_done = 0
        results = []

    logger.info("Loaded %d remaining claims", len(df))

    for idx, row in df.iterrows():
        logger.info(
            "Processing claim %d/%d (user_id=%s)",
            idx + 1,
            already_done + len(df),
            row.get("user_id"),
        )

        try:
            output_row = process_row(row)

        except Exception as e:
            logger.error("Row %d failed entirely: %s", idx, e)

            output_row = {col: None for col in OUTPUT_COLUMNS}
            output_row["user_id"] = row.get("user_id")
            output_row["image_paths"] = row.get("image_paths")
            output_row["user_claim"] = row.get("user_claim")
            output_row["claim_object"] = row.get("claim_object")

        results.append(output_row)

        # AUTOSAVE AFTER EVERY CLAIM
        pd.DataFrame(
            results,
            columns=OUTPUT_COLUMNS
        ).to_csv(
            OUTPUT_CSV,
            index=False,
            quoting=csv.QUOTE_MINIMAL
        )

    logger.info(
    "Finished. Wrote %d rows to %s",
    len(results),
    OUTPUT_CSV
)

if __name__ == "__main__":
    main()