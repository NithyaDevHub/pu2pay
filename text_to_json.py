
import json
import os
from paddleocr import PaddleOCR
import torch

from transformers import pipeline
import re

from db import get_db_connection
from json_output.get_value_in_object import get_key_value_pairs_in_order

pipe = pipeline("text-generation", model="./zephyr-7b-alpha", torch_dtype=torch.bfloat16)

IMAGE_DIR = "/Users/fis/Documents/pu2pay/invoices"

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Helper to check if file is an image
def is_image_file(file_path):
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))

# OCR extraction
def extract_text(image_path):
    result = ocr.ocr(image_path, cls=True)
    if result and isinstance(result[0], list):
        return " ".join([line[1][0] for line in result[0]])

def Image_to_JSON(image_name, extract_text, pdf_ref_id):

    messages = [
   {
  "role": "system",
    "content": "You are a general JSON converter for OCR information, organizing the information into a structured JSON output.",
},
{
  "role": "user",
  "content": f'Extract the following fields: "invoice_number", "invoice_date", "total_amount", "po_ref", "company_name","supplier_name", "supplier_address","bill_to_company_name","bill_to_address","bill_to_state","bill_to_state_code","bill_to_pan","bill_to_cin", "ship_to_company_name", "ship_to_address", "ship_to_state", "ship_to_state_code", "ship_to_gstin", "ship_to_pan", "ship_to_cin", "pan_supplier", "gstin_supplier","bill_to_gstin", "udyam_regno", "state", "state_code",  "invoice_id", "discount", "Qty", "total_taxable_amount", "total_discount_value", "total_quantity", "taxable_value", "tcs_rate", "total_cgst_amount", "total_sgst_amount", "total_tax_amount", "total_invoice_value", "line_items.item_name", "line_items.hsn", "line_items.item_qty", "line_items.uom", "line_items.rate_incl_of_tax", "line_items.unit_price", "line_items.total_retail_price", "line_items.total_taxable_amount","line_items.discount", "line_items.total_value", {extract_text}',
#   "content": f'Extract the following fields and groupd them into the following 1. Invoice Details, 2. Bill_to, 3. Ship_to, 4. Line Items, 5.Inovice Summary, others", {extract_text}',
},]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=2500, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
    generated_text = outputs[0]["generated_text"]
    
    output_folder = (
    "/Users/fis/Documents/pu2pay/json_output"
    )
    
    # output_folder = "diagnostic_report_summary_output" if classification == "DiagnosticReports" else "kyc_report_output"  if classification == "KycReports" else "cheque_report_output"  if classification == "ChequeReports"   else "discharge_summary_output"
    print("output folder", output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    txt_filename = f"{image_name}.txt"
    txt_path = os.path.join(output_folder, txt_filename)
    print("generated_text", generated_text)
    with open(txt_path, 'w') as file:
        # print(file,'--------')
        file.write(generated_text)
        print(f"Saved {txt_filename} to {output_folder}")
    
    clean_text = re.sub(r'<.*?>', '', generated_text)
    json_start = clean_text.find('{')
    json_end = clean_text.rfind('}') + 1   
    json_string = clean_text[json_start:json_end]
    print("json_string", json_string)
    json_data = json.loads(json_string)
    invoice_check_list = get_key_value_pairs_in_order(json_data)
    print("invoice_check_list", invoice_check_list)
    print("json_data ---------------------------------", json_data)

    search_keys = ["invoice_number", "invoice_date","vendor_name","address","supplier_gstin","buyer_gstin","supplier_pan","total_invoice_value","gstin_pan"
                ]

    results = search_nested(json_string, search_keys)
    
    insert_invoice_duplicate_check(results,pdf_ref_id)
    print("Matching keys found:")
    for k, v in results:
        print(f"Key: {k}, Value: {v}")
    return txt_path



def search_nested(data, keys_to_find):
    matches = []

    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in keys_to_find:
                    matches.append((key, value))
                if isinstance(value, (dict, list)):
                    recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(data)
    return matches

def insert_invoice_duplicate_check(data, pdf_ref_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO invoice_duplicates_check (
                pdf_ref_id, invoice_number, invoice_date, vendor_name, address,
                supplier_gstin, buyer_gstin, supplier_pan, gstin_pan,
                invoice_sum_amount_total_amount, is_duplicate
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            pdf_ref_id,
            data.get("invoice_number"),
            data.get("invoice_date"),
            data.get("vendor_name"),
            data.get("address"),
            data.get("supplier_gstin"),
            data.get("buyer_gstin"),
            data.get("supplier_pan"),
            data.get("gstin_pan"),
            data.get("total_invoice_value"),
            True
        ))
        conn.commit()
        print("✅ Duplicate check record inserted.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error inserting duplicate check: {e}")
    finally:
        cursor.close()
        conn.close()