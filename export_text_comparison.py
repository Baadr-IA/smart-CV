import os
from pathlib import Path
from unittest.mock import patch
from outils.parser import _parse_pdf_native, _ocr_pdf, _clean_text
from bench_ocr import _prepare_noisy_pdf

def dummy_preprocess(img):
    return img

def run_export():
    pdf_path = Path("input/cv-developpeur-web.pdf")
    if not pdf_path.exists():
        print("Fichier introuvable.")
        return

    output_dir = Path("debug_text_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # 1. RÉFÉRENCE (Ground Truth de PyPDF)
    gt_raw, _ = _parse_pdf_native(pdf_path)
    gt_text = _clean_text(gt_raw)
    with open(output_dir / "0_GROUND_TRUTH_PYPDF.txt", "w", encoding="utf-8") as f:
        f.write(gt_text)

    noise_levels = ["none", "medium"]
    
    for noise in noise_levels:
        print(f"Traitement bruit : {noise}")
        noisy_pdf = _prepare_noisy_pdf(pdf_path, noise, Path("bench_output_metrics/noisy_pdfs"))

        # 2. SANS OPENCV (Tesseract Brut)
        with patch('outils.image_processor.preprocess_for_ocr', side_effect=dummy_preprocess):
            with patch('outils.image_processor.get_text_blocks', return_value=[]):
                txt_sans = _clean_text(_ocr_pdf(noisy_pdf))
                with open(output_dir / f"TEXT_SANS_OPENCV_{noise}.txt", "w", encoding="utf-8") as f:
                    f.write(txt_sans)

        # 3. AVEC TON CODE (Nettoyage + Segmentation)
        txt_avec = _clean_text(_ocr_pdf(noisy_pdf))
        with open(output_dir / f"TEXT_AVEC_OPENCV_{noise}.txt", "w", encoding="utf-8") as f:
            f.write(txt_avec)

    print(f"\nTerminé ! Regarde dans le dossier '{output_dir}'.")
    print("Compare 'TEXT_AVEC_OPENCV_none.txt' avec '0_GROUND_TRUTH_PYPDF.txt' pour voir l'ordre.")
    print("Regarde 'TEXT_AVEC_OPENCV_medium.txt' pour voir si les mots sont là malgré le CER de 220%.")

if __name__ == "__main__":
    run_export()
