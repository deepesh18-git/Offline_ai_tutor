 #pdf_reader.py

import os
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    all_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            all_text.append(page_text)

    full_text = "\n".join(all_text)
    print(f"Extracted {len(full_text)} characters from {pdf_path}")
    return full_text


def save_raw_text(text, filename, output_dir="data/raw_text"):

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path