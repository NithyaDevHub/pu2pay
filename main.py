import shutil
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Body, UploadFile
from typing import List, Dict
from fastapi.staticfiles import StaticFiles
import psycopg2
import psycopg2.extras
from duplicate import compare_with_reference_image
from models import *
import json
import os
import torch
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Body
from CRUD.invoice import insert_invoice
from db import get_db_connection
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_path
import uuid


app = FastAPI(title="PU2PAY API - Invoice, PO, MRN, RAO")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins like ["http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow specific methods like ["GET", "POST"]
    allow_headers=["*"],  # Allow specific headers
)
app.mount("/images", StaticFiles(directory="/Users/fis/Documents/pu2pay/output_convert_images"), name="images")
app.mount("/images_duplicate", StaticFiles(directory="/Users/fis/Documents/pu2pay/dublicate_output_images"), name="images_duplicate")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the LayoutLMv2 model and processor
model_path = '/Users/fis/Documents/claim_summerisation/LayoutLMv2_models/LayoutLMv2_06092024.h5'
class_names_path = '/Users/fis/Documents/claim_summerisation/LayoutLMv2_models/class_names.json'

class FolderPathRequest(BaseModel):
    root_input: str

# def load_model_and_processor(model_path, class_names_path):
#     try:
#         model = LayoutLMv2ForSequenceClassification.from_pretrained(model_path).to(device)
#         processor = LayoutLMv2Processor.from_pretrained('microsoft/layoutlmv2-base-uncased')
#         with open(class_names_path, 'r') as f:
#             class_names = json.load(f)
#         return model, processor, class_names
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error loading model or processor: {e}")


# model, processor, class_names = load_model_and_processor(model_path, class_names_path)


# ------------------- Convert PDFs to Images -------------------


async def process_folder(request: FolderPathRequest):
    try:
        root_input = request.root_input
        root_output = os.path.join(os.getcwd(), "/Users/fis/Documents/pu2pay/output_images")
        
        if os.path.exists(root_output):
            shutil.rmtree(root_output)
        os.makedirs(root_output, exist_ok=True)
        # return
        print("step 1 started ------------>")
        start_time = datetime.now()
        # Step 1: Convert PDFs to images
        pdf_conversion_result = process_pdfs(root_input, root_output)
        if not pdf_conversion_result:
            raise HTTPException(status_code=400, detail="PDF conversion failed or no PDFs found.")
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

        return {
            "message": "Process completed successfully."
           
        }


    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during the process: {str(e)}")


def process_pdfs(root_input, root_output, overwrite=True):
    results = []
    total_pdfs = sum([len([f for f in files if f.lower().endswith('.pdf')]) for r, d, files in os.walk(root_input)])
    print(total_pdfs, "---------------------------")
    print("root_input", root_input)
    
    if total_pdfs == 0:
        raise HTTPException(status_code=400, detail="No PDFs found in the input folder.")

    for dirpath, dirnames, filenames in os.walk(root_input):
        rel_path = os.path.relpath(dirpath, root_input)
        output_path = os.path.join(root_output, rel_path)
        os.makedirs(output_path, exist_ok=True)

        pdf_files = [f for f in filenames if f.lower().endswith('.pdf')]
        num_pdfs = len(pdf_files)
        num_images = 0
        status = "Processed"
        insert_id = None
        # Database insertion (wrapped in try to avoid crashing the loop)
        try:
            # insert_pdf_conversion(os.path.basename(dirpath), num_pdfs, num_images, status)
            # try:
            insert_id = insert_pdf_conversion(os.path.basename(dirpath), num_pdfs, num_images, status)
            print(f"Inserted DB record ID: {insert_id}")
        except Exception as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Database error: {e}")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(dirpath, pdf_file)
            pdf_filename = os.path.splitext(pdf_file)[0]

            try:
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images, start=1):
                    image_filename = f"{pdf_filename}_{i}.png"
                    image_path = os.path.abspath(os.path.join(output_path, image_filename))
                    print("image_path", image_path)
                    print("pdf_path", pdf_path)
                    # "/Users/fis/Documents/pu2pay/output_convert_images"
                   
                    # Only overwrite if the flag is True, otherwise skip if the image exists
                    if os.path.exists(image_path) and not overwrite:
                        continue

                    image.save(image_path, "PNG")
                    reference_images = compare_with_reference_image(image_path, "/Users/fis/Documents/pu2pay/output_convert_images",insert_id,image_filename)
                    print("+++++++++++++++++++++++++++",reference_images)
                    num_images += 1
                     # ✅ If no reference found, move the image
                    if not reference_images:
                        convert_output_dir = "/Users/fis/Documents/pu2pay/output_convert_images"
                        os.makedirs(convert_output_dir, exist_ok=True)
                        destination_path = os.path.join(convert_output_dir, os.path.basename(image_path))
                        
                        shutil.move(image_path, destination_path)
                        print(f"Moved unreferenced image to: {destination_path}")
                    else:
                        convert_output_dir = "/Users/fis/Documents/pu2pay/dublicate_output_images"
                        os.makedirs(convert_output_dir, exist_ok=True)
                        destination_path = os.path.join(convert_output_dir, os.path.basename(image_path))
                        
                        shutil.move(image_path, destination_path)
                        print(f"Moved unreferenced image to: {destination_path}")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                status = "Error"
                break

        

        results.append({
            "Claim_id": os.path.basename(dirpath),  # Using folder name as Claim ID
            "Number of PDFs": num_pdfs,
            "Number of Images": num_images,
            "Status": status
        })

    return results

def process_images(root_input, root_output, overwrite=True):
    results = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    total_images = sum([len([f for f in files if f.lower().endswith(supported_extensions)]) for r, d, files in os.walk(root_input)])
    print(f"Total image files: {total_images}")
    
    if total_images == 0:
        raise HTTPException(status_code=400, detail="No image files found in the input folder.")

    for dirpath, dirnames, filenames in os.walk(root_input):
        rel_path = os.path.relpath(dirpath, root_input)
        output_path = os.path.join(root_output, rel_path)
        os.makedirs(output_path, exist_ok=True)

        image_files = [f for f in filenames if f.lower().endswith(supported_extensions)]
        num_images = 0
        status = "Processed"
        insert_id = None
        # reference_images_1 = None
        try:
            insert_id = insert_pdf_conversion(os.path.basename(dirpath), 0, len(image_files), status)
            print(f"Inserted DB record ID for images: {insert_id}")
        except Exception as e:
            print(f"Database error: {e}")

        for image_file in image_files:
            image_path = os.path.join(dirpath, image_file)
            image_filename = os.path.splitext(image_file)[0] + ".png"  # Normalize extension
            normalized_path = os.path.abspath(os.path.join(output_path, image_filename))

            try:
                if not overwrite and os.path.exists(normalized_path):
                    continue

                # Convert and save normalized copy in output_path
                image = Image.open(image_path)
                image.save(normalized_path, "PNG")  # Save to output_path

                # Compare and decide where to copy the file
                # reference_images = None
                reference_images = compare_with_reference_image(
                    normalized_path,
                    "/Users/fis/Documents/pu2pay/output_convert_images",
                    insert_id,
                    image_filename
                )
                print("+++++++++++++++++++++++++++", reference_images)
                num_images += 1
                reference_images_1 = reference_images
                if not reference_images:
                    convert_output_dir = "/Users/fis/Documents/pu2pay/output_convert_images"
                else:
                    convert_output_dir = "/Users/fis/Documents/pu2pay/dublicate_output_images"

                os.makedirs(convert_output_dir, exist_ok=True)
                destination_path = os.path.join(convert_output_dir, os.path.basename(normalized_path))
                
                # ⬇️ Copy instead of move so that image also stays in output_path
                shutil.copy(normalized_path, destination_path)
                print(f"Copied image to: {destination_path}")

            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                status = "Error"
                break

        results.append({
            "Claim_id": os.path.basename(dirpath),
            "Number of PDFs": 0,
            "Number of Images": num_images,
            "Status": status
            # "image_duplicates": reference_images_1
        })

    return results


def insert_pdf_conversion(subfolder, num_pdfs, num_images, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO pdf_conversion_hypotus (Claim_id, num_pdfs, num_images, status)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (subfolder, int(num_pdfs), int(num_images), status)
        )
        inserted_id = cursor.fetchone()[0]
        conn.commit()
        return inserted_id
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error inserting PDF conversion data: {str(e)}")
    finally:
        cursor.close()
        conn.close()
        
        
    


TABLE_MODEL_MAPPING = {
    "invoice_details": InvoiceDetails,
    "invoice_supplier_details": InvoiceSupplierDetails,
    "invoice_buyer_details": InvoiceBuyerDetails,
    "invoice_shipping_details": InvoiceShippingDetails,
    "invoice_lineitems": InvoiceLineItems,
    "invoice_summary": InvoiceSummary,
    "invoice_bank_details": InvoiceBankDetails,
    "invoice_asset_details": InvoiceAssetDetails,
    "purchase_order_details": PurchaseOrderDetails,
    "po_details" :POOrderDetails,
    "po_supplier_details": POSupplierDetails,
    "po_buyer_details": POBillTo,
    "po_shipping_details": POShippingDetails,
    "po_lineitems": POLineItems,
    "po_summary": POSummary,
    "po_terms_conditions": POTermsConditions,
    "mrn_details": MRNDetails,
    "mrn_supplier_details": MRNSupplierDetails,
    "mrn_buyer_details": MRNBuyerDetails,
    "mrn_lineitems": MRNLineItems,
    "mrn_summary": MRNSummary,
    "rao_details": RAODetails,
    "rao_supplier_details": RAOSupplierDetails,
    "rao_receivedat": RAOBuyerDetails,
    "rao_line_items": RAOLineItems,
    "rao_summary": RAOSummary,
    "invoice_po_match_results": InvoicePOMatchResult,
    "invoice_po_summary": InvoicePOSummary
}

ALLOWED_EXTENSIONS = (".pdf",)
ALLOWED_IMAGES = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# def allowed_file(filename: str) -> bool:
#     ext = os.path.splitext(filename)[1].lower()
#     return ext in ALLOWED_EXTENSIONS
def allowed_file(filename):
    return filename.lower().endswith(ALLOWED_EXTENSIONS + ALLOWED_IMAGES)

def generate_batch_folder_name() -> str:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")  # Format: 2025-05-14_10-43-20
    return f"BATCH_{date_str}"

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF and image files are allowed.")

    # Create folder with current timestamp
    folder_name = generate_batch_folder_name()
    folder_path = os.path.join(UPLOAD_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save file to disk
    file_location = os.path.join(folder_path, file.filename)
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    print("Saved File:", file_location)

    try:
        root_input = folder_path
        root_output = os.path.join(os.getcwd(), "output_images")
        
        if os.path.exists(root_output):
            shutil.rmtree(root_output)
        os.makedirs(root_output, exist_ok=True)

        print("Processing started ------------>")
        start_time = datetime.now()

        # Decide which method to run
        if file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            process_result = process_pdfs(root_input, root_output)
            process_type = "PDF conversion"
        else:
            process_result = process_images(root_input, root_output)
            process_type = "Image processing"

        if not process_result:
            raise HTTPException(status_code=400, detail=f"{process_type} failed or no valid files found.")

        end_time = datetime.now()
        print(f"{process_type} completed in: {end_time - start_time}")

        return JSONResponse(
            status_code=200,
            content={
                "folder": folder_name,
                "filename": file.filename,
                "message": f"{process_type} completed successfully."
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# async def upload_file(file: UploadFile = File(...)):
#     if not allowed_file(file.filename):
#         raise HTTPException(status_code=400, detail="Only PDF and image files are allowed.")

#     # Create folder with current date and time
#     folder_name = generate_batch_folder_name()
#     folder_path = os.path.join(UPLOAD_DIR, folder_name)
#     os.makedirs(folder_path, exist_ok=True)

#     # Save file in that folder
#     file_location = os.path.join(folder_path, file.filename)
#     # process_folder(file_location)
    
#     with open(file_location, "wb") as f:
#         content = await file.read()
#         f.write(content)
#     print("+++++++++++",file_location)
#     try:
#         root_input = folder_path
#         root_output = os.path.join(os.getcwd(), "/Users/fis/Documents/pu2pay/output_images")
        
#         if os.path.exists(root_output):
#             shutil.rmtree(root_output)
#         os.makedirs(root_output, exist_ok=True)
#         # return
#         print("step 1 started ------------>")
#         start_time = datetime.now()
#         # Step 1: Convert PDFs to images
#         pdf_conversion_result = process_pdfs(root_input, root_output)
#         if not pdf_conversion_result:
#             raise HTTPException(status_code=400, detail="PDF conversion failed or no PDFs found.")
#         end_time = datetime.now()
#         print('Duration: {}'.format(end_time - start_time))

#         return JSONResponse(
#         status_code=200,
#         content={
#             "folder": folder_name,
#             "filename": file.filename,
#             "message": "File uploaded successfully."
#         }
#     )


    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred during the process: {str(e)}")
    
    
    
# --- Insert Record Endpoint ---
@app.post("/insert/{table_name}")
def insert_record(table_name: str, record: dict):
    if table_name not in TABLE_MODEL_MAPPING:
        raise HTTPException(status_code=400, detail=f"Table {table_name} not supported yet.")

    model = TABLE_MODEL_MAPPING[table_name]
    try:
        validated_data = model(**record)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    data = validated_data.dict()
    columns = data.keys()
    values = data.values()

    insert_query = f"""
    INSERT INTO {table_name} ({','.join(columns)})
    VALUES ({','.join(['%s'] * len(columns))})
    RETURNING id;
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(insert_query, list(values))
        inserted_id = cur.fetchone()[0]
        conn.commit()
        return {"message": f"Record inserted into {table_name}", "id": inserted_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# --- Get Records by Batch ID ---
@app.get("/get/{table_name}/{batch_id}")
def get_records_by_batch_id(table_name: str, batch_id: str):
    if table_name not in TABLE_MODEL_MAPPING:
        raise HTTPException(status_code=400, detail=f"Table {table_name} not supported yet.")

    query = f"SELECT * FROM {table_name} WHERE batch_id = %s ORDER BY id ASC;"

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, (batch_id,))
        records = cur.fetchall()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()


@app.get("/get_records_by_po_id/{table_name}/{po_id}")
def get_records_by_po_id(table_name: str, po_id: str):
    if table_name not in TABLE_MODEL_MAPPING:
        raise HTTPException(status_code=400, detail=f"Table {table_name} not supported yet.")

    query = f"SELECT * FROM {table_name} WHERE po_id = %s ORDER BY po_id ASC;"

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, (po_id,))
        records = cur.fetchall()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()


@app.get("/get_records_by_invoice_no/{table_name}/{invoice_id}")
def get_records_by_po_id(table_name: str, invoice_id: str):
    if table_name not in TABLE_MODEL_MAPPING:
        raise HTTPException(status_code=400, detail=f"Table {table_name} not supported yet.")

    query = f"SELECT * FROM {table_name} WHERE invoice_id = %s ORDER BY invoice_id ASC;"

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, (invoice_id,))
        records = cur.fetchall()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()


@app.get("/get_records_by_grn_no/{table_name}/{grn_id}")
def get_records_by_po_id(table_name: str, grn_id: str):
    if table_name not in TABLE_MODEL_MAPPING:
        raise HTTPException(status_code=400, detail=f"Table {table_name} not supported yet.")

    query = f"SELECT * FROM {table_name} WHERE grn_id = %s ORDER BY grn_id ASC;"

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, (grn_id,))
        records = cur.fetchall()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()
        
      
        
@app.get("/get_invoice_full_details/{invoice_number}")
def get_invoice_full_details(invoice_number: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Fetch invoice_details
        cur.execute("SELECT * FROM invoice_details WHERE invoice_number = %s", (invoice_number,))
        invoice = cur.fetchone()
        if not invoice:
            raise HTTPException(status_code=404, detail="Invoice not found")

        invoice_id = invoice["invoice_id"]  # Adjust if your column is named 'id'

        # Fetch from child tables
        def fetch_child(table_name):
            cur.execute(f"SELECT * FROM {table_name} WHERE invoice_id = %s", (invoice_id,))
            return cur.fetchall()

        supplier_details = fetch_child("invoice_supplier_details")
        buyer_details = fetch_child("invoice_buyer_details")
        shipping_details = fetch_child("invoice_shipping_details")
        line_items = fetch_child("invoice_lineitems")
        asset_details = fetch_child("invoice_asset_details")
        bank_details = fetch_child("invoice_bank_details")
        summary = fetch_child("invoice_summary")

        return {
            "invoice_details": invoice,
            "supplier_details": supplier_details,
            "buyer_details": buyer_details,
            "shipping_details": shipping_details,
            "line_items": line_items,
            "asset_details": asset_details,
            "bank_details": bank_details,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
            
            
            
@app.get("/get_po_full_details/{po_number}")
def get_invoice_full_details(po_number: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Fetch invoice_details
        cur.execute("SELECT * FROM po_details WHERE po_number = %s", (po_number,))
        invoice = cur.fetchone()
        if not invoice:
            raise HTTPException(status_code=404, detail="Invoice not found")

        invoice_id = invoice["po_id"]  # Adjust if your column is named 'id'

        # Fetch from child tables
        def fetch_child(table_name):
            cur.execute(f"SELECT * FROM {table_name} WHERE po_id = %s", (invoice_id,))
            return cur.fetchall()

        supplier_details = fetch_child("po_supplier_details")
        buyer_details = fetch_child("po_billto")
        shipping_details = fetch_child("po_shipping_details")
        line_items = fetch_child("po_lineitems")
        summary = fetch_child("po_summary")
        tc = fetch_child("po_terms_conditions")

        return {
            "po_details": invoice,
            "supplier_details": supplier_details,
            "buyer_details": buyer_details,
            "shipping_details": shipping_details,
            "line_items": line_items,
            "terms_and_conditions": tc,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
            
            
            
@app.get("/get_mrn_full_details/{mrn_number}")
def get_invoice_full_details(mrn_number: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Fetch invoice_details
        cur.execute("SELECT * FROM mrn_details WHERE mrn_number = %s", (mrn_number,))
        mrn = cur.fetchone()
        if not mrn:
            raise HTTPException(status_code=404, detail="Invoice not found")

        mrn_id = mrn["id"]  # Adjust if your column is named 'id'

        # Fetch from child tables
        def fetch_child(table_name):
            cur.execute(f"SELECT * FROM {table_name} WHERE mrn_id = %s", (mrn_id,))
            return cur.fetchall()

        supplier_details = fetch_child("mrn_supplier_details")
        buyer_details = fetch_child("mrn_buyer_details")
        line_items = fetch_child("mrn_lineitems")
        summary = fetch_child("mrn_summary")

        return {
            "mrn_details": mrn,
            "supplier_details": supplier_details,
            "buyer_details": buyer_details,
            "line_items": line_items,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
            
            
@app.get("/get_rao_full_details/{rao_number}")
def get_invoice_full_details(rao_number: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Fetch invoice_details
        cur.execute("SELECT * FROM rao_details WHERE rao_number = %s", (rao_number,))
        rao = cur.fetchone()
        if not rao:
            raise HTTPException(status_code=404, detail="Invoice not found")

        rao_id = rao["rao_id"]  # Adjust if your column is named 'id'

        # Fetch from child tables
        def fetch_child(table_name):
            cur.execute(f"SELECT * FROM {table_name} WHERE rao_id = %s", (rao_id,))
            return cur.fetchall()

        supplier_details = fetch_child("rao_supplier_details")
        buyer_details = fetch_child("rao_receivedat")
        line_items = fetch_child("rao_lineitems")
        summary = fetch_child("rao_summary")

        return {
            "rao_details": rao,
            "supplier_details": supplier_details,
            "buyer_details": buyer_details,
            "line_items": line_items,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
            
            
@app.get("/get_conversion_details")
async def get_pdf_conversion_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM pdf_conversion_hypotus;"
        cursor.execute(query)
        records = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        result = [dict(zip(colnames, record)) for record in records]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image_classification data: {str(e)}")
    finally:
        cursor.close()
        conn.close()
        
@app.get("/classification_details")
async def get_classification_details_table_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM image_classification_hypotus;"
        cursor.execute(query)
        records = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        result = [dict(zip(colnames, record)) for record in records]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image_classification data: {str(e)}")
    finally:
        cursor.close()
        conn.close()
        

@app.get("/get_claim_id_classification_details/{claim_id}")
async def get_classification_details_table_data(claim_id:str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM image_classification_hypotus WHERE claim_id = '"+claim_id+"';"
        cursor.execute(query)
        records = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        result = [dict(zip(colnames, record)) for record in records]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image_classification data: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/get_bbox/{top_label}")
async def get_bbox(top_label:str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM hypotus_bbox WHERE image_path = '"+top_label+"';"
        cursor.execute(query)
        records = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        result = [dict(zip(colnames, record)) for record in records]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image_classification data: {str(e)}")
    finally:
        cursor.close()
        conn.close()
        
        

@app.post("/insert_invoice_full_details")
def insert_invoice_full_details(payload: Dict = Body(...)):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        invoice = payload["invoice_details"]
        invoice_number = invoice["invoice_number"]

        # Insert into invoice_details
        cur.execute("""
            INSERT INTO invoice_details (
                invoice_number, invoice_date, total_amount, bill_to, bill_to_address,
                ship_to, ship_to_address, po_ref
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING invoice_id
        """, (
            invoice["invoice_number"],
            invoice["invoice_date"],
            invoice["total_amount"],
            invoice["bill_to"],
            invoice["bill_to_address"],
            invoice["ship_to"],
            invoice["ship_to_address"],
            invoice["po_ref"]
        ))
        invoice_id = cur.fetchone()[0]

        # Insert into all child tables
        def insert_child(table, rows):
            for row in rows:
                row["invoice_id"] = invoice_id  # attach FK
                cols = ", ".join(row.keys())
                vals = ", ".join(["%s"] * len(row))
                cur.execute(f"INSERT INTO {table} ({cols}) VALUES ({vals})", tuple(row.values()))

        insert_child("invoice_supplier_details", payload.get("supplier_details", []))
        insert_child("invoice_buyer_details", payload.get("buyer_details", []))
        insert_child("invoice_shipping_details", payload.get("shipping_details", []))
        insert_child("invoice_lineitems", payload.get("line_items", []))
        insert_child("invoice_asset_details", payload.get("asset_details", []))
        insert_child("invoice_bank_details", payload.get("bank_details", []))
        insert_child("invoice_summary", payload.get("summary", []))

        conn.commit()
        return {"message": "Invoice inserted successfully", "invoice_id": invoice_id}

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.put("/update_invoice_full_details/{invoice_number}")
def update_invoice_full_details(invoice_number: str, payload: Dict = Body(...)):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get invoice_id
        cur.execute("SELECT invoice_id FROM invoice_details WHERE invoice_number = %s", (invoice_number,))
        result = cur.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Invoice not found")
        invoice_id = result[0]

        # Update invoice_details
        invoice = payload["invoice_details"]
        cur.execute("""
            UPDATE invoice_details SET
                invoice_date = %s,
            WHERE invoice_number = %s
        """, (
            invoice["invoice_date"],
            invoice_number
        ))

        # Delete old child records
        for table in [
            "invoice_supplier_details", "invoice_buyer_details", "invoice_shipping_details",
            "invoice_lineitems", "invoice_asset_details", "invoice_bank_details", "invoice_summary"
        ]:
            cur.execute(f"DELETE FROM {table} WHERE invoice_id = %s", (invoice_id,))

        # Insert new child records
        def insert_child(table, rows):
            for row in rows:
                row["invoice_id"] = invoice_id
                cols = ", ".join(row.keys())
                vals = ", ".join(["%s"] * len(row))
                cur.execute(f"INSERT INTO {table} ({cols}) VALUES ({vals})", tuple(row.values()))

        insert_child("invoice_supplier_details", payload.get("supplier_details", []))
        insert_child("invoice_buyer_details", payload.get("buyer_details", []))
        insert_child("invoice_shipping_details", payload.get("shipping_details", []))
        insert_child("invoice_lineitems", payload.get("line_items", []))
        insert_child("invoice_asset_details", payload.get("asset_details", []))
        insert_child("invoice_bank_details", payload.get("bank_details", []))
        insert_child("invoice_summary", payload.get("summary", []))

        conn.commit()
        return {"message": f"Invoice {invoice_number} updated successfully"}

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/get_invoice_po_match/{invoice_id}", response_model=List[InvoicePOMatchResult])
def get_invoice_po_match(invoice_id: int):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM get_invoice_po_match_results_v2(%s);", (invoice_id,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

        cur.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No data found.")

        results = [dict(zip(columns, row)) for row in rows]
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
@app.get("/get_invoice_po_summary/{invoice_id}", response_model=List[InvoicePOSummary])
def get_invoice_po_summary(invoice_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM get_invoice_po_summary(%s);", (invoice_id,))
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]

        cursor.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No summary data found.")

        result = [dict(zip(col_names, row)) for row in rows]
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    


@app.get("/invoice_po_details")
def get_invoice_po_details():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM get_invoice_with_po_details();")
        result = cur.fetchall()
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
            
            
            
@app.get("/invoice_po_details/{invoice_id}")
def get_invoice_po_details(invoice_id: int):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT * FROM get_invoice_with_po_details_by_id(%s)", (invoice_id,))
        result = cur.fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Invoice not found")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()
        
        
# @app.get("/3way_match_record", response_model=List[ThreeWayMatchRecord])
# # @app.get("/api/3way-match", response_model=List[MatchRecord])
# def get_3way_matching_report():
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#         cur.execute("SELECT * FROM vw_3way_matching_report;")
#         rows = cur.fetchall()
#         columns = [desc[0] for desc in cur.description]

#         cur.close()
#         conn.close()

#         # Check sample row data
#         if not rows:
#             return []

#         for row in rows[:2]:
#             print("Sample row:", row)

#         # result = [dict(zip(columns, row)) for row in rows]
#         return rows

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_3_way_match_checklist/{inv_id}")
def get_lineitem_summary(inv_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM get_3_way_checklist(%s)", (inv_id,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    result = [dict(zip(columns, row)) for row in rows]
    cur.close()
    conn.close()
    return result


@app.get("/get_metadata_checklist/{inv_id}")
def get_metadata_checklist(inv_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM get_metadata_checklist(%s)", (inv_id,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    result = [dict(zip(columns, row)) for row in rows]
    cur.close()
    conn.close()
    return result


@app.get("/combined_transaction_summary/{invoice_id}", response_model=CombinedSummary)
def get_combined_summary(invoice_id: int):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM get_combined_transaction_summary(%s);", (invoice_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Invoice not found or no summary available.")

        return row

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image_duplicates/{id}")
def image_duplicates(id: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # ✅ Wrap id in a tuple
        cur.execute("SELECT * FROM image_duplicates WHERE pdf_ref_id = %s;", (id,))

        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Image duplicate not found for the given ID.")

        return row

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/insert_invoice")
def insert_invoice_api(data: InvoiceData):
    try:
        invoice_id = insert_invoice(data.invoice_details, data.line_items)
        return {"message": "Inserted successfully", "invoice_id": invoice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def update_pdf_status(id_value, new_status):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE public.pdf_conversion_hypotus
            SET status = %s
            WHERE id = %s
        """, (new_status, id_value))
        conn.commit()
        print(f"Updated status for ID {id_value} to '{new_status}'")
    except Exception as e:
        conn.rollback()
        print(f"Error updating status: {e}")
    finally:
        cursor.close()
        conn.close()
