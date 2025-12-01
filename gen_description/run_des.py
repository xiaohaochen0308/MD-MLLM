#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_descri.py

Batch-generate semantic descriptions for images using an OpenAI-compatible
multimodal model, and save them into a CSV file.

Main features:
- Reads image list from a CSV (expects at least `id` and `annotation` columns)
- Constructs image path as: root_dir / annotation / id
- Calls a vision-language model to generate a short description (caption)
- Supports resumable processing (if output CSV already exists)
- Batch writes to CSV and retries failed rows

You can adapt this script to your own environment by modifying the config
section below (paths, column names, model name, prompt, etc.).
"""

import csv
import os
import base64
import time

from openai import OpenAI
from tqdm import tqdm


# =============================
#  OpenAI-compatible client
# =============================

# If you use the official OpenAI API, you can simply do:
#   client = OpenAI(api_key="YOUR_API_KEY")
# and remove base_url below.
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Change to your own endpoint
    api_key="YOUR_API_KEY"                # Or "EMPTY" if not required
)


def load_image_base64(path):
    """Read an image from disk and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_api(image_path):
    """
    Call the multimodal chat.completions API to generate an image description.

    NOTE:
    - The return format of `message.content` may depend on your server
      implementation. Here we assume it is directly a string.
    - If your server returns a list (OpenAI v1 style), you may need to change:
        return response.choices[0].message.content[0].text
    """
    img_b64 = load_image_base64(image_path)

    # You can customize the prompt as needed.
    prompt = "Give semantic descriptions of the image in 50 words."

    response = client.chat.completions.create(
        # Replace with your model name or path
        model="Qwen2.5-VL-7B-Instruct",
        max_tokens=50,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


# =============================
#  Config (edit for your use case)
# =============================

# Input CSV file. It should contain at least:
# - "id": image file name (e.g., "xxx.jpg", "xxx.png")
# - "annotation": subfolder name or some category used in the path
input_csv = "input.csv"

# Output CSV file. This script supports resumable processing:
# if this file already exists, existing rows will be reused.
output_csv = "output_with_captions.csv"

# Root directory of images. The script will look for images at:
# root_dir / annotation / id
root_dir = "images_root"

# New caption column name to be added to the CSV.
caption_column = "mllm_caption"

# Flush to disk every N rows (for large datasets).
batch_size = 100


# =============================
#  Step 1: Load existing output (resumable)
# =============================

existing = {}
if os.path.exists(output_csv):
    print(f"[INFO] Loading existing output: {output_csv}")
    with open(output_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Use "id" as the unique key
            existing[r["id"]] = r

# Read input CSV
with open(input_csv, "r", newline="") as f:
    reader = list(csv.DictReader(f))
    # Build fieldnames: if caption column already exists, don't duplicate
    fieldnames = list(reader[0].keys())
    if caption_column not in fieldnames:
        fieldnames.append(caption_column)


# =============================
#  Step 2: Process with batch write
# =============================

buffer = []
failed_rows = []

print("\n===== Processing =====")
for row in tqdm(reader, desc="Processing images"):
    img_id = row["id"]

    # If caption already exists in existing output, reuse that row (resumable)
    if img_id in existing and existing[img_id].get(caption_column):
        buffer.append(existing[img_id])
    else:
        img_path = os.path.join(root_dir, row["annotation"], img_id)

        if not os.path.exists(img_path):
            print(f"[WARN] Missing: {img_path}")
            row[caption_column] = ""
            failed_rows.append(row)
        else:
            try:
                caption = call_api(img_path)
                row[caption_column] = caption
            except Exception as e:
                print(f"[ERROR] {img_id} failed: {e}")
                row[caption_column] = ""
                failed_rows.append(row)

        buffer.append(row)

    # Batch write: flush every `batch_size` rows
    if len(buffer) >= batch_size:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(buffer)
        print(f"[INFO] Saved {len(buffer)} rows (batch write)")


# Final write after the loop
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(buffer)

print("\n===== First Pass Done =====")
print(f"Total: {len(buffer)}")
print(f"Failed: {len(failed_rows)}")


# =============================
#  Step 3: Retry failed rows
# =============================

print(f"\n===== Retrying {len(failed_rows)} failed rows =====")
retry_success = 0

for row in tqdm(failed_rows, desc="Retrying"):
    img_path = os.path.join(root_dir, row["annotation"], row["id"])

    if not os.path.exists(img_path):
        continue

    ok = False
    for _ in range(3):
        try:
            caption = call_api(img_path)
            row[caption_column] = caption
            retry_success += 1
            ok = True
            break
        except Exception:
            time.sleep(0.5)

    # Update the corresponding row in buffer
    if ok:
        for i, r in enumerate(buffer):
            if r["id"] == row["id"]:
                buffer[i] = row
                break

# Final write after retry
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(buffer)

print("\n===== DONE =====")
print(f"Recovered: {retry_success}")
print(f"Output saved: {output_csv}")
