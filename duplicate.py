import os
import cv2
import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from paddleocr import PaddleOCR
from PIL import Image
import imagehash


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model


from db import get_db_connection
from text_to_json import Image_to_JSON

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Define paths
IMAGE_DIR = "/Users/fis/Documents/pu2pay/output_convert_images"
# IMAGE_DIR = "/Users/fis/Documents/pu2pay/invoices"


# Helper to check if file is an image
def is_image_file(file_path):
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))

# OCR extraction
def extract_text(image_path):
    result = ocr.ocr(image_path, cls=True)
    if result and isinstance(result[0], list):
        return " ".join([line[1][0] for line in result[0]])
    return ""

# Image hash
def compute_image_hash(image_path):
    img = Image.open(image_path)
    return str(imagehash.phash(img))

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def compute_feature_vector(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.flatten()

# Text similarity
def text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

# ELA-based tampering detection (auto deletes ELA file)
def detect_tampering(image_path):
    ela_image_path = image_path.replace(".png", "_ela.png")
    img = Image.open(image_path).convert('RGB')
    img.save(ela_image_path, 'JPEG', quality=90)
    temp_img = Image.open(ela_image_path)
    ela_img = Image.blend(img, temp_img, alpha=0.5)
    ela_array = np.array(ela_img) - np.array(temp_img)
    tampering_score = np.mean(ela_array)

    # Auto-delete ELA image
    try:
        if os.path.exists(ela_image_path):
            os.remove(ela_image_path)
    except Exception as e:
        print(f"Warning: Failed to delete ELA image: {ela_image_path} - {e}")

    return tampering_score

# Compare reference image to all in the folder
def compare_with_reference_image(reference_image_path, image_dir, insert_id,image_filename):
    reference_text = extract_text(reference_image_path)
    print("Reference text:", insert_id)
    

    reference_hash = compute_image_hash(reference_image_path)
    reference_vector = compute_feature_vector(reference_image_path)
    reference_tampering_score = detect_tampering(reference_image_path)

    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        duplicate_file_path = os.path.join("/Users/fis/Documents/pu2pay/dublicate_output_images", file_name)
        # duplicate_file_path = os.path.join("", file_name)
        if not os.path.isfile(file_path) or not is_image_file(file_path):
            continue

        try:
            target_text = extract_text(file_path)
            target_hash = compute_image_hash(file_path)
            target_vector = compute_feature_vector(file_path)
            target_tampering_score = detect_tampering(file_path)

            feature_sim = cosine_similarity([reference_vector], [target_vector])[0][0]
            text_sim = text_similarity(reference_text, target_text)
            hash_match = reference_hash == target_hash

            duplicate_status = "Exact Duplicate" if hash_match else (
                "Near Duplicate" if feature_sim >= 0.90 and text_sim >= 0.90 else "Not Duplicate"
            )

            if duplicate_status != "Not Duplicate":
                result = {
                    "Reference_Image": os.path.basename(reference_image_path),
                    "Target_Image": file_name,
                    "Feature_Similarity": round(float(feature_sim), 3),
                    "Text_Similarity": round(float(text_sim), 3),
                    "Hash_Match": hash_match,
                    "Reference_Tampering_Score": round(float(reference_tampering_score), 2),
                    "Target_Tampering_Score": round(float(target_tampering_score), 2),
                    "Duplicate_Status": duplicate_status
                }

                with open("duplicate_result.json", "w") as f:
                    json.dump(result, f, indent=4)

                print(f"{duplicate_status} found and saved to 'duplicate_result.json'")
                
                # Insert only exact duplicates into DB
                if duplicate_status != "Not Duplicate":
                    print("Inserting into database...",result)
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute("""
                            INSERT INTO image_duplicates (
                                reference_image, target_image, feature_similarity,
                                text_similarity, reference_tampering_score,
                                target_tampering_score, duplicate_status, pdf_ref_id
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            result["Reference_Image"],
                            result["Target_Image"],
                            result["Feature_Similarity"],
                            result["Text_Similarity"],
                            result["Reference_Tampering_Score"],
                            result["Target_Tampering_Score"],
                            result["Duplicate_Status"],
                            insert_id
                        ))
                        if duplicate_status != "Not Duplicate":
                            cur.execute("""
                                UPDATE pdf_conversion_hypotus
                                SET status = %s
                                WHERE id = %s
                            """, ("Exact Duplicate Found", insert_id))

                        conn.commit()
                        cur.close()
                        conn.close()
                        print("Record inserted and status updated if applicable.")

                        return result
                    except Exception as e:
                        print(f"Inserting issues {file_path}: {e}")
            # else : 
            #     print(f"Not a duplicate: {file_name} - {duplicate_status}")
            #     Image_to_JSON(image_filename, reference_text,insert_id)
        except Exception as e:
            print(f"Error comparing {file_path}: {e}")

    print("No duplicate or near duplicate found.")
    return None
# Run comparison
# compare_with_reference_image(REFERENCE_IMAGE, IMAGE_DIR)
