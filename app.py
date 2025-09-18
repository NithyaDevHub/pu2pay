import os
import re
import json
import torch
import paddle
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature


# ------------------ OCR & LLM Service ------------------
class OCRService:
    ocr_model = None
    models = {}
    gpu_handle = None

    @classmethod
    def init_gpu_monitor(cls):
        try:
            nvmlInit()
            cls.gpu_handle = nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f"[WARNING] NVML GPU monitoring failed: {e}")
            cls.gpu_handle = None

    @classmethod
    def log_gpu(cls):
        if cls.gpu_handle:
            mem = nvmlDeviceGetMemoryInfo(cls.gpu_handle)
            util = nvmlDeviceGetUtilizationRates(cls.gpu_handle)
            temp = nvmlDeviceGetTemperature(cls.gpu_handle, 0)
            print(
                f"[GPU] Memory: {mem.used // (1024 ** 2)}/{mem.total // (1024 ** 2)} MB "
                f"({mem.used / mem.total * 100:.1f}%) | Util: {util.gpu}% | Temp: {temp}°C"
            )

    @classmethod
    def load_ocr(cls):
        if cls.ocr_model is None:
            try:
                paddle.set_device("cpu")  # Always CPU for OCR
                cls.ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
                print("[OK] PaddleOCR loaded on CPU ✅")
            except Exception as e:
                print(f"[ERROR] Failed to load PaddleOCR: {e}")

    @classmethod
    def load_all_models(cls):
        cls.init_gpu_monitor()
        base_path = r"F:\invoice_extraction"
        model_map = {"zephyr": "zephyr-7b-alpha"}

        for key, subdir in model_map.items():
            full_path = os.path.join(base_path, subdir)
            if not os.path.exists(full_path):
                print(f"[WARNING] Model path not found: {full_path}")
                continue

            print(f"[INIT] Loading model '{key}' on GPU...")
            tokenizer = AutoTokenizer.from_pretrained(full_path, use_fast=True, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                full_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True,
                quantization_config=bnb_config
            )
            model.eval()

            # Optional compilation
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                except Exception as e:
                    print(f"[WARNING] Model compilation failed: {e}")

            cls.models[key] = {"tokenizer": tokenizer, "model": model}
            print(f"[OK] Model '{key}' loaded ✅")
            cls.log_gpu()

    @classmethod
    def get_model(cls, model_name):
        return cls.models.get(model_name)

    @classmethod
    def get_ocr(cls):
        if cls.ocr_model is None:
            cls.load_ocr()
        return cls.ocr_model


# ------------------ JSON Extractor ------------------
def extract_json_from_response(text: str):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        return None
    json_str = json_match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


def generate_json(contract_text: str):
    model_data = OCRService.get_model("zephyr")
    if not model_data:
        raise ValueError("Zephyr model not loaded")

    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    system_message = "You are a JSON converter. Return ONLY valid JSON, no other text."
    user_message = f"""
    You are an AI that extracts contract details. 

    Task:
    From the given contract text below, extract **only** the information that fits into the following JSON keys. 
    - Return **strictly** this JSON structure.
    - Do **not** add extra fields, nested objects, or explanations.
    - If a value is not present, leave it as an empty string ("").

    Required JSON format:

    {{
      "First Party": "",
      "Second Party":"",
      "Contract Type": ""(can be Sales Contract,Service Contract,Construction Contract,Lease Agreement,Employment Contract,Partnership Agreement,Non-Disclosure Agreement (NDA),Loan Agreement,Franchise Agreement,Consultancy Agreement
),
      "Services": "",
      "Service Amount": "",
      "Contract Duration": "",
      "Contract Start Date": "",
      "Contract Expiry Date": "",
      "Jurisdiction (Which Court – Location)": "",
      "Signatories (Who All Signed the Contract)": "",
      "Payment Terms and Conditions": "",
      "Penalties (If Any)": ""
    }}

    Contract Text:
    {contract_text}

    JSON:
    """

    prompt = f"<|system|>\n{system_message}\n<|user|>\n{user_message}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=True
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"full_response -------- \n {full_response}")
    generated_text = full_response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
    print(f"generated_text -------- \n {generated_text}")
    json_data = extract_json_from_response(generated_text)
    print(f"json_data -------- \n {json_data}")
    if not json_data:
        raise ValueError("Failed to extract valid JSON")
    return json_data


# ------------------ Process Folder ------------------
def process_contract_folder(folder_path: str):
    ocr = OCRService.get_ocr()
    full_text = ""

    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not img_files:
        raise ValueError(f"No images found in folder {folder_path}")

    for img_file in img_files:
        # print(f"processing ocr for ---- {img_file}")
        img_path = os.path.join(folder_path, img_file)
        try:
            results = ocr.ocr(img_path)

            # ✅ Your OCR extraction logic
            if results and isinstance(results[0], dict):
                ocr_lines = results[0].get("rec_texts", [])
            else:
                ocr_lines = [line[1][0] for line in results[0]] if results else []

            ocr_text = "\n".join(ocr_lines)

            if not ocr_text.strip():
                print(f"[Warning] No text found in {img_file}")
            else:
                print(f"[OK] Extracted text from {img_file}")

            full_text += ocr_text + "\n"

        except Exception as e:
            print(f"[ERROR] OCR failed for {img_file}: {e}")

    if not full_text.strip():
        raise ValueError("No text extracted from folder")

    return full_text

if __name__ == "__main__":
    # Load PaddleOCR (CPU)
    OCRService.load_ocr()
    # Load Zephyr LLM (GPU)
    OCRService.load_all_models()

    folder_path = r"F:\contract_docs\images\IN91-070225-PR-028"
    # folder_path = r"F:\contract_docs\images\IN91-070225-PR-017"
    # folder_path = r"F:\contract_docs\images\IN91-070125-PR-101"
    # folder_path = r"F:\contract_docs\images\IN91-070125-PR-099"
    # folder_path = r"F:\contract_docs\images\IN91-070125-PR-097"

    contract_text = process_contract_folder(folder_path)

    json_data = generate_json(contract_text)
    print("\n✅ Final JSON Output:\n")
    print(json.dumps(json_data, indent=2, ensure_ascii=False))
