import os
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# Load API Key
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")

# Load Image
img = Image.open("dataset/images/sample/case_001/img_1.jpg")

# Prompt
prompt = """
You are an insurance damage reviewer.

Analyze this image.

Return ONLY valid JSON.

{
  "object_type":"",
  "issue_type":"",
  "object_part":"",
  "damage_visible":true,
  "severity":"",
  "quality_flags":[]
}

Allowed object_type:
car
laptop
package

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

Allowed severity:
none
low
medium
high
unknown

Allowed quality_flags:
blurry_image
cropped_or_obstructed
low_light_or_glare
wrong_angle
damage_not_visible

If there are no quality problems return:
[]
"""

# Send to Gemini
response = model.generate_content([
    prompt,
    img
])

# Print Result
print("\n===== GEMINI RESPONSE =====\n")
print(response.text)