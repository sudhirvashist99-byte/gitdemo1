import os
import subprocess

INPUT_PDF_DIR = "/mydata/input_pdfs"
IMAGE_DIR = "/mydata/work_images"

os.makedirs(IMAGE_DIR, exist_ok=True)

for pdf in os.listdir(INPUT_PDF_DIR):
    if not pdf.lower().endswith(".pdf"):
        continue

    name = os.path.splitext(pdf)[0]
    out_dir = os.path.join(IMAGE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    pdf_path = os.path.join(INPUT_PDF_DIR, pdf)

    subprocess.run([
        "pdftoppm",
        "-png",
        "-r", "300",
        pdf_path,
        os.path.join(out_dir, "page")
    ])

print("PDF → Images completed")
