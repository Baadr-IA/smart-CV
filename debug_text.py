
from pathlib import Path
import os
from outils.parser import _parse_pdf_native, _ocr_pdf, _clean_text

def export_debug_text():
    # On utilise le fichier qui pose problème (82% CER)
    pdf_path = Path("input/cvsofianedevlast.pdf")
    if not pdf_path.exists():
        print(f"Fichier introuvable : {pdf_path}")
        return

    print(f"Analyse de : {pdf_path}")

    # 1. Extraction Native
    print("Extraction native (PyPDF)...")
    try:
        native_raw, _ = _parse_pdf_native(pdf_path)
        native_text = _clean_text(native_raw)
        with open("debug_native.txt", "w", encoding="utf-8") as f:
            f.write(native_text)
        print(" -> 'debug_native.txt' créé.")
    except Exception as e:
        print(f" Erreur native : {e}")

    # 2. Extraction OCR
    print("Extraction OCR (Tesseract + OpenCV)...")
    try:
        ocr_raw = _ocr_pdf(pdf_path)
        ocr_text = _clean_text(ocr_raw)
        with open("debug_ocr.txt", "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(" -> 'debug_ocr.txt' créé.")
    except Exception as e:
        print(f" Erreur OCR : {e}")

if __name__ == "__main__":
    export_debug_text()
