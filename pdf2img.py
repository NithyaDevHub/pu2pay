import os
import fitz  # PyMuPDF

def convert_pdfs_to_images(input_folder, output_folder, dpi=300):
    """
    Convert all PDFs in input_folder (recursively) to images.
    For each PDF, create a subfolder with the PDF name and save page images inside.
    """

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                pdf_name = os.path.splitext(file)[0]  # filename without .pdf

                # Maintain folder structure + PDF subfolder
                rel_path = os.path.relpath(root, input_folder)
                save_dir = os.path.join(output_folder, rel_path, pdf_name)
                os.makedirs(save_dir, exist_ok=True)

                try:
                    doc = fitz.open(pdf_path)

                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        zoom = dpi / 72  # scale based on DPI
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)

                        img_filename = f"{pdf_name}_page{page_num+1}.png"
                        img_path = os.path.join(save_dir, img_filename)
                        pix.save(img_path)

                    doc.close()
                    print(f"✅ Converted: {pdf_path}")

                except Exception as e:
                    print(f"❌ Error processing {pdf_path}: {e}")


# if __name__ == "__main__":
#     input_folder = r"input_pdfs"     # folder with PDFs
#     output_folder = r"output_images" # output images
#     convert_pdfs_to_images(input_folder, output_folder)

if __name__ == "__main__":
    input_folder = r"F:\contract_docs\raw"
    output_folder = r"F:\contract_docs\images"
    convert_pdfs_to_images(input_folder, output_folder)
