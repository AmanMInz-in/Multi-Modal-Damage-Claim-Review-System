import os
import json
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
import google.generativeai as genai

# Load API key
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

# Read claims
claims = pd.read_csv("dataset/claims.csv")

# Take first claim
row = claims.iloc[0]

print("\nUSER CLAIM:")
print(row["user_claim"])

print("\nIMAGE PATHS:")
print(row["image_paths"])

# Load all images
image_paths = row["image_paths"].split(";")

images = []

for path in image_paths:
    full_path = os.path.join("dataset", path)

    if os.path.exists(full_path):
        images.append(Image.open(full_path))

print(f"\nLoaded {len(images)} images")

prompt = f"""
You are an insurance claim reviewer.

User claim:
{row["user_claim"]}

Analyze ALL images together.

Return ONLY valid JSON.

{{
  "issue_type":"",
  "object_part":"",
  "damage_visible":true,
  "severity":"",
  "supporting_image_ids":[]
}}

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

Use only allowed car parts:
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

Allowed severity:
none
low
medium
high
unknown
"""

response = model.generate_content(
    [prompt] + images
)

print("\n===== ANALYSIS =====\n")
print(response.text)