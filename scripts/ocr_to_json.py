import os
import json
from paddleocr import PaddleOCR

IMAGE_ROOT = "/mydata/work_images"
OUTPUT_ROOT = "/mydata/output_json"

ocr = PaddleOCR(lang="en", use_gpu=False)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for doc in os.listdir(IMAGE_ROOT):
    doc_path = os.path.join(IMAGE_ROOT, doc)
    if not os.path.isdir(doc_path):
        continue

    out_doc = os.path.join(OUTPUT_ROOT, doc)
    os.makedirs(out_doc, exist_ok=True)

    for img in sorted(os.listdir(doc_path)):
        if not img.endswith(".png"):
            continue

        img_path = os.path.join(doc_path, img)
        result = ocr.ocr(img_path, cls=False)

        page_json = {
            "image": img,
            "lines": []
        }

        if result and result[0]:
            for line in result[0]:
                page_json["lines"].append({
                    "text": line[1][0],
                    "confidence": float(line[1][1]),
                    "box": line[0]
                })

        with open(os.path.join(out_doc, img.replace(".png", ".json")), "w") as f:
            json.dump(page_json, f, indent=2)

print("OCR → JSON completed")

