import os
import time
import difflib
from pathlib import Path
from unittest.mock import patch

from outils.parser import _parse_pdf_native, _ocr_pdf, _clean_text
from bench_ocr import _prepare_noisy_pdf

def dummy_preprocess(img):
    return img

import re

def normalize_string(s):
    """Normalisation pour une comparaison sémantique juste."""
    s = s.lower()
    # Supprimer les caractères spéciaux/puces en début de ligne
    s = re.sub(r'^[^a-z0-9]+', '', s)
    # Remplacer tout ce qui n'est pas alphanumérique par un espace
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # Condenser les espaces multiples en un seul
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def calculate_line_metrics(gt_text, ocr_text, threshold=0.7):
    """
    Calcule Precision, Recall et F1-Score avec normalisation des chaînes.
    """
    # Nettoyage, découpage et normalisation
    gt_lines = [normalize_string(l) for l in gt_text.split('\n') if len(l.strip()) > 3]
    ocr_lines = [normalize_string(l) for l in ocr_text.split('\n') if len(l.strip()) > 3]
    
    # On enlève les lignes qui sont devenues vides après normalisation
    gt_lines = [l for l in gt_lines if l]
    ocr_lines = [l for l in ocr_lines if l]
    
    if not gt_lines: return 0.0, 0.0, 0.0
    if not ocr_lines: return 0.0, 0.0, 0.0

    matched_gt = set()
    tp = 0 # True Positives (Lignes trouvées)

    for o_idx, o_line in enumerate(ocr_lines):
        best_ratio = 0
        best_gt_idx = -1
        
        for g_idx, g_line in enumerate(gt_lines):
            if g_idx in matched_gt: continue
            
            # Similarité de séquence (Levenshtein normalisé)
            ratio = difflib.SequenceMatcher(None, o_line, g_line).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_gt_idx = g_idx
        
        # Si la ligne ressemble à > 70% à une ligne de la vérité terrain
        if best_ratio >= threshold:
            tp += 1
            matched_gt.add(best_gt_idx)

    precision = tp / len(ocr_lines) if len(ocr_lines) > 0 else 0.0
    recall = tp / len(gt_lines) if len(gt_lines) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate_metrics():
    input_dir = Path("input")
    pdfs = [f for f in input_dir.iterdir() if f.suffix.lower() == ".pdf"]
    
    dump_dir = Path("DUMP_RESULTATS_F1")
    dump_dir.mkdir(exist_ok=True)

    noise_levels = ["none", "medium"]
    results = []

    print(f"Évaluation du F1-Score par Ligne sur {len(pdfs)} CVs...")

    for pdf in pdfs:
        print(f"\n--- Analyse de {pdf.name} ---")
        gt_raw, _ = _parse_pdf_native(pdf)
        gt_text = _clean_text(gt_raw)
        if len(gt_text) < 150: continue

        for noise in noise_levels:
            noisy_pdf = _prepare_noisy_pdf(pdf, noise, Path("bench_output_metrics/noisy_pdfs"))
            
            # 1. SANS PRÉTRAITEMENT NI SEGMENTATION
            with patch('outils.image_processor.preprocess_for_ocr', side_effect=dummy_preprocess):
                with patch('outils.image_processor.get_text_blocks', return_value=[]):
                    txt_sans = _clean_text(_ocr_pdf(noisy_pdf))
                    p_s, r_s, f1_s = calculate_line_metrics(gt_text, txt_sans)

            # 2. AVEC TON PIPELINE COMPLET
            txt_avec = _clean_text(_ocr_pdf(noisy_pdf))
            p_a, r_a, f1_a = calculate_line_metrics(gt_text, txt_avec)

            print(f"  [Bruit {noise.upper()}]")
            print(f"    - SANS OpenCV : F1={f1_s:.2%} (P={p_s:.2%}, R={r_s:.2%})")
            print(f"    - AVEC OpenCV : F1={f1_a:.2%} (P={p_a:.2%}, R={r_a:.2%})")
            
            results.append({
                "noise": noise,
                "f1_sans": f1_s, "f1_avec": f1_a,
                "p_sans": p_s, "p_avec": p_a,
                "r_sans": r_s, "r_avec": r_a
            })

    print("\n=================================================")
    print("BILAN FINAL : PERFORMANCE INGÉNIEUR (F1-SCORE)")
    print("=================================================")
    for noise in noise_levels:
        res = [r for r in results if r["noise"] == noise]
        if not res: continue
        avg_f1_no = sum(r["f1_sans"] for r in res) / len(res)
        avg_f1_yes = sum(r["f1_avec"] for r in res) / len(res)
        avg_p_yes = sum(r["p_avec"] for r in res) / len(res)
        avg_r_yes = sum(r["r_avec"] for r in res) / len(res)
        
        print(f"\nConditions {noise.upper()}:")
        print(f"  F1-Score Moyen (AVEC OpenCV) : {avg_f1_yes:.2%}")
        print(f"  -> Précision (Bruit) : {avg_p_yes:.2%}")
        print(f"  -> Rappel (Structure) : {avg_r_yes:.2%}")
        print(f"  Gain par rapport au brut : {avg_f1_yes - avg_f1_no:+.2%}")

if __name__ == '__main__':
    evaluate_metrics()
