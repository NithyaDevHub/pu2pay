import os
import re
import json
import time
import torch
import warnings
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
from mlx_vlm import load, apply_chat_template, generate

# ==========================
# 🔧 Configuration
# ==========================
INPUT_FOLDER = "/Users/fis/Documents/Data/Showri_POCs/ACRDemo2"
OUTPUT_FOLDER = "./govind_data/vlm_output_ACRDemo2"
CSV_SUMMARY_PATH = "./govind_data/invoice_summary_vlm_output_ACRDemo2.csv"
EXCEL_SUMMARY_PATH = "./govind_data/invoice_summary_vlm_output_ACRDemo2.xlsx"
RATE_PER_1000_TOKENS_INR = 5.00

EXPECTED_FIELDS = [
    "Invoice No", "Invoice Date", "Seller GSTIN", "Seller Pan", "Seller Name",
    "Buyer GSTIN", "Buyer Name", "Buyer Pan", "Ship to GSTIN", "Ship to Name",
    "Subtotal Amount", "Discount Amount", "CGST Amount", "SGST Amount",
    "IGST Amount", "Cess Amount", "Additianal Cess Amount"
]

PROMPT = f"""
You are an invoice field extractor. Extract ONLY the following fields from the image and return a JSON object with EXACTLY these keys:

{chr(10).join([f"- {field}" for field in EXPECTED_FIELDS])}

Format the output exactly like this:
{{
  "Invoice No": {{"value": "...", "bbox": [x_min, y_min, x_max, y_max], "confidence": 0.95}},
  ...
}}
Ensure bounding boxes are in absolute pixel coordinates and return confidence scores for each field. Do not include explanations or additional text.
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================
# 📊 Utilities
# ==========================
def extract_json(image, prompt, model, processor, config):
    try:
        prompt_filled = apply_chat_template(processor, config, prompt)
        raw_output, _ = generate(model, processor, prompt_filled, image, max_tokens=2048, temperature=0)
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        json_data = json.loads(match.group()) if match else {}
        tokens = len(prompt_filled.split()) + len(raw_output.split())
        return json_data, raw_output, tokens
    except Exception as e:
        print(f"⚠️ Warning: Failed to process inputs with error: {str(e)}")
        return {}, "", 0

def normalize_fields(json_data):
    for field in EXPECTED_FIELDS:
        if field not in json_data or not isinstance(json_data[field], dict):
            json_data[field] = {"value": "N/A", "bbox": [0, 0, 0, 0], "confidence": 0.0}
        else:
            json_data[field].setdefault("value", "N/A")
            json_data[field].setdefault("bbox", [0, 0, 0, 0])
            json_data[field].setdefault("confidence", 0.0)
    return json_data

# ==========================
# 🤖 Worker (Serial)
# ==========================
def process_file(filepath, output_dir, model, processor, config):
    try:
        start_time = datetime.now()
        ext = filepath.suffix.lower()
        base_name = filepath.stem
        folder_name = filepath.parent.name

        if ext == ".pdf":
            images = convert_from_path(str(filepath), dpi=300)
        else:
            images = [Image.open(filepath).convert("RGB")]

        all_rows = []

        for page_idx, image in enumerate(images):
            json_data, raw_output, tokens = extract_json(image, PROMPT, model, processor, config)
            json_data = normalize_fields(json_data)

            page_suffix = f"_page{page_idx+1}" if len(images) > 1 else ""
            image_output_path = output_dir / f"{base_name}{page_suffix}.jpg"
            json_output_path = output_dir / f"{base_name}{page_suffix}.json"

            image.save(image_output_path)
            with open(json_output_path, "w") as f:
                json.dump(json_data, f, indent=2)

            print(f"\n📄 Extracted JSON for {filepath.name}{page_suffix}:")
            print(json.dumps(json_data, indent=2))
            print("-" * 60)

            for field in EXPECTED_FIELDS:
                props = json_data[field]
                all_rows.append({
                    "File Name": filepath.name,
                    "Folder Name": folder_name,
                    "Page": page_idx + 1,
                    "Field": field,
                    "Value": props.get("value"),
                    "Confidence": props.get("confidence"),
                    "BBox": props.get("bbox"),
                    "Tokens Used": tokens,
                    "INR Cost": round(tokens / 1000 * RATE_PER_1000_TOKENS_INR, 2),
                    "Start Time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "End Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        return all_rows

    except Exception as e:
        print(f"❌ Error processing {filepath.name}: {e}")
        return [{"File Name": filepath.name, "Folder Name": filepath.parent.name, "Page": 1, "Field": field, "Value": "ERROR", "Confidence": 0.0, "BBox": [0, 0, 0, 0], "Tokens Used": 0, "INR Cost": 0.0, "Start Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "End Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for field in EXPECTED_FIELDS]

# ==========================
# 🚀 Run
# ==========================
def run_pipeline(retry_mode=False):
    start_all = time.time()
    all_rows = []

    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"
    model, processor = load(model_name)
    config = model.config

    if retry_mode:
        with open(output_path / "failed_files.txt") as f:
            retry_files = [Path(line.strip()) for line in f.readlines() if line.strip()]
        files = retry_files
        print(f"🔁 Retrying {len(files)} failed files...")
    else:
        files = [Path(root) / file for root, _, files in os.walk(input_path)
                 for file in files if file.lower().endswith((".pdf", ".jpg", ".jpeg", ".png"))]

    print(f"🔍 Found {len(files)} files to process...")

    for i, file in enumerate(files):
        print(f"➡️ [{i+1}/{len(files)}] Processing: {file.name}")
        rows = process_file(file, output_path, model, processor, config)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_SUMMARY_PATH, index=False)

    pivot = df.pivot_table(index=["File Name", "Folder Name", "Page"], columns="Field", values="Value", aggfunc="first").reset_index()
    pivot.to_csv(output_path / "pivot_invoice_summary.csv", index=False)

    summary_fields = df.pivot_table(index=["File Name", "Folder Name", "Page"], values="Value", columns="Field", aggfunc="first").reset_index()
    summary_fields.to_csv(output_path / "field_level_summary.csv", index=False)

    cost_summary = df.groupby("File Name").agg({
        "Tokens Used": "sum",
        "INR Cost": "sum",
        "Start Time": "first",
        "End Time": "last"
    }).reset_index()

    cost_summary = cost_summary.sort_values(by="INR Cost", ascending=False)

    grand_total = pd.DataFrame.from_records([{
        "File Name": "TOTAL",
        "Tokens Used": df["Tokens Used"].sum(),
        "INR Cost": df["INR Cost"].sum(),
        "Start Time": "",
        "End Time": ""
    }])

    cost_summary = pd.concat([cost_summary, grand_total], ignore_index=True)
    cost_summary.to_csv(output_path / "cost_and_time_summary.csv", index=False)

    # Save to Excel workbook
    with pd.ExcelWriter(EXCEL_SUMMARY_PATH, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="All_Fields", index=False)
        pivot.to_excel(writer, sheet_name="Pivot_Summary", index=False)
        summary_fields.to_excel(writer, sheet_name="Field_Level_Summary", index=False)
        cost_summary.to_excel(writer, sheet_name="Cost_Summary", index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    cost_chart = cost_summary[cost_summary["File Name"] != "TOTAL"]
    plt.barh(cost_chart["File Name"], cost_chart["INR Cost"], color="skyblue")
    plt.xlabel("INR Cost")
    plt.ylabel("File Name")
    plt.title("Token Cost per File")
    plt.tight_layout()
    plt.savefig(output_path / "cost_chart.png")

    failed = df[df["Value"] == "ERROR"]["File Name"].unique()
    if len(failed):
        with open(output_path / "failed_files.txt", "w") as f:
            for fname in failed:
                f.write(fname + "\n")

    print("\n😁 Processing Complete")
    print(f"🧠 Total Tokens: {df['Tokens Used'].sum()}")
    print(f"💰 Total Price: ₹{df['INR Cost'].sum():.2f}")
    print(f"⏱️ Time Taken: {round(time.time() - start_all, 2)} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry", action="store_true", help="Retry failed files only")
    args = parser.parse_args()

    run_pipeline(retry_mode=args.retry)
