import os
import json
from paddleocr import PaddleOCR

INPUT_DIR = "/data/images"
OUTPUT_DIR = "/data/output_json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ocr = PaddleOCR(lang="en", cls=False)

for img in sorted(os.listdir(INPUT_DIR)):
    if not img.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, img)
    result = ocr.ocr(img_path)

    output = {
        "image": img,
        "text_blocks": []
    }

    for line in result[0]:
        output["text_blocks"].append({
            "text": line[1][0],
            "confidence": float(line[1][1]),
            "box": line[0]
        })

    out_file = os.path.join(
        OUTPUT_DIR,
        img.rsplit(".", 1)[0] + ".json"
    )

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"OCR done → {out_file}")
