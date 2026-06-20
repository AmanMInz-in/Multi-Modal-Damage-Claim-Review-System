import os
import json
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
import google.generativeai as genai

# Load API Key
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

# Load sample data
sample = pd.read_csv("dataset/sample_claims.csv")

# First sample
row = sample.iloc[0]

# Load images
image_paths = row["image_paths"].split(";")

images = []

for path in image_paths:
    full_path = os.path.join("dataset", path)
    images.append(Image.open(full_path))

prompt = f"""
You are an insurance damage claim reviewer.

USER CLAIM:
{row["user_claim"]}

Analyze all submitted images.

IMPORTANT RULES:

1. Return ONLY ONE issue_type.
2. Return ONLY ONE object_part.
3. Use ONLY values from the allowed lists.
4. Do not invent extra damage.
5. Focus on the primary damage most relevant to the user's claim.
6. Severity must be one of:
   none, low, medium, high, unknown

Allowed issue_type:
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

Allowed car object_part:
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

Return ONLY valid JSON:

{{
  "issue_type":"",
  "object_part":"",
  "severity":"",
  "damage_visible":true
}}
"""

response = model.generate_content(
    [prompt] + images
)

# Clean response
text = response.text
text = text.replace("```json", "")
text = text.replace("```", "")
text = text.strip()

# Parse JSON
prediction = json.loads(text)

print("\n========== EXPECTED ==========\n")

print("issue_type :", row["issue_type"])
print("object_part:", row["object_part"])
print("severity   :", row["severity"])

print("\n========== PREDICTED ==========\n")

print("issue_type :", prediction["issue_type"])
print("object_part:", prediction["object_part"])
print("severity   :", prediction["severity"])