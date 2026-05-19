"""
Benchmark AVANT/APRÈS prétraitement OpenCV v2
Métriques : CER, WER, Jaccard-mots, BLEU-4
4 CVs × 2 niveaux de bruit (light / medium)
"""
import sys
import os
import re
import math
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from outils.parser import _parse_pdf_native, _ocr_pdf, _clean_text
from outils.image_processor import preprocess_for_ocr as _orig_preprocess

# ── Paramètres ──────────────────────────────────────────────────────────────
NOISY_DIR  = Path("bench_output_metrics/noisy_pdfs")
INPUT_DIR  = Path("input")
DUMP_DIR   = Path("DUMP_RESULTATS_TEXTE")           # met à jour les dumps AVEC_OPENCV
DUMP_DIR.mkdir(exist_ok=True)

CVS = [
    "cv-developpeur-web",
    "cvsofianedevlast",
    "Exemple de CV Full stack developer",
    "fake_cv",
]
NOISE_LEVELS = ["light", "medium"]

# CVs multi-colonnes : deskew désactivé pour préserver le layout
NO_DESKEW = {"cvsofianedevlast"}

# ── Helpers métriques ────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def edit_distance(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j]+1, curr[j-1]+1, prev[j-1]+(0 if ca==cb else 1)))
        prev = curr
    return prev[-1]

def cer(ref: str, hyp: str) -> float:
    r, h = list(normalize(ref)), list(normalize(hyp))
    if not r: return 1.0
    return min(edit_distance(r, h) / len(r), 2.0)  # cap à 2.0

def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r: return 1.0
    return min(edit_distance(r, h) / len(r), 2.0)

def jaccard(ref: str, hyp: str) -> float:
    r, h = set(normalize(ref).split()), set(normalize(hyp).split())
    if not r and not h: return 1.0
    return len(r & h) / len(r | h) if (r | h) else 0.0

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_n(ref_tokens, hyp_tokens, n):
    """Précision n-gram clippée (composant BLEU)."""
    if len(hyp_tokens) < n: return 0.0
    ref_ng = {}
    for ng in ngrams(ref_tokens, n):
        ref_ng[ng] = ref_ng.get(ng, 0) + 1
    matches = 0
    for ng in ngrams(hyp_tokens, n):
        if ref_ng.get(ng, 0) > 0:
            matches += 1
            ref_ng[ng] -= 1
    return matches / max(len(hyp_tokens) - n + 1, 1)

def bleu4(ref: str, hyp: str) -> float:
    r = normalize(ref).split()
    h = normalize(hyp).split()
    if not r or not h: return 0.0
    # Brevity penalty
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r)/len(h))
    precisions = [bleu_n(r, h, n) for n in range(1, 5)]
    if any(p == 0 for p in precisions): return 0.0
    log_avg = sum(math.log(p) for p in precisions) / 4
    return round(bp * math.exp(log_avg), 4)

def calc_all(ref: str, hyp: str) -> dict:
    return {
        "CER":    round(cer(ref, hyp), 4),
        "WER":    round(wer(ref, hyp), 4),
        "Jaccard": round(jaccard(ref, hyp), 4),
        "BLEU-4": bleu4(ref, hyp),
    }

# ── Wrapper OCR ──────────────────────────────────────────────────────────────
def dummy_preprocess(img, deskew=True):
    """Bypass prétraitement OpenCV."""
    return img

def preprocess_no_deskew(img, deskew=True):
    """Pipeline v2 sans deskew (pour CV multi-colonnes)."""
    return _orig_preprocess(img, deskew=False)

def run_ocr_no_seg(pdf_path: Path, preprocess_fn=None) -> str:
    """
    OCR Tesseract page entière (psm 1), avec ou sans preprocessing.
    Bypass la segmentation par blocs pour une comparaison strictement isolée
    sur l'effet du prétraitement (seule variable).
    """
    if preprocess_fn is not None:
        with patch('outils.image_processor.preprocess_for_ocr', side_effect=preprocess_fn):
            with patch('outils.image_processor.get_text_blocks', return_value=[]):
                return _clean_text(_ocr_pdf(pdf_path))
    else:
        with patch('outils.image_processor.get_text_blocks', return_value=[]):
            return _clean_text(_ocr_pdf(pdf_path))

def run_sans(pdf_path: Path) -> str:
    """OCR Tesseract brut, sans aucun prétraitement (seule variable : pas de preprocessing)."""
    return run_ocr_no_seg(pdf_path, preprocess_fn=dummy_preprocess)

def run_avec(pdf_path: Path, cv_stem: str) -> str:
    """OCR Tesseract avec pipeline v2 — même segmentation que SANS (page entière)."""
    if cv_stem in NO_DESKEW:
        return run_ocr_no_seg(pdf_path, preprocess_fn=preprocess_no_deskew)
    return run_ocr_no_seg(pdf_path, preprocess_fn=None)  # utilise la vraie preprocess_for_ocr

# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    results = []

    print("=" * 72)
    print("  BENCHMARK AVANT/APRÈS PRÉTRAITEMENT v2 (NLM + AdaptiveThreshold)")
    print("=" * 72)

    for cv_stem in CVS:
        original_pdf = INPUT_DIR / f"{cv_stem}.pdf"
        if not original_pdf.exists():
            print(f"\n⚠  {cv_stem}.pdf introuvable dans input/ — ignoré")
            continue

        # Ground truth via pypdf
        gt_raw, _ = _parse_pdf_native(original_pdf)
        gt_text = _clean_text(gt_raw)
        if len(gt_text) < 100:
            print(f"\n⚠  Ground truth trop court pour {cv_stem} — ignoré")
            continue

        # Sauvegarde GT mise à jour
        (DUMP_DIR / f"{cv_stem}_0_GROUND_TRUTH.txt").write_text(gt_text, encoding="utf-8")

        print(f"\n{'─'*72}")
        print(f"  CV : {cv_stem}")
        print(f"{'─'*72}")
        print(f"  {'Condition':<35} {'CER':>7} {'WER':>7} {'Jaccard':>8} {'BLEU-4':>8}")
        print(f"  {'─'*35} {'─'*7} {'─'*7} {'─'*8} {'─'*8}")

        for noise in NOISE_LEVELS:
            noisy_pdf = NOISY_DIR / f"{cv_stem}__{noise}.pdf"
            if not noisy_pdf.exists():
                print(f"  ⚠  PDF bruit {noise} introuvable ({noisy_pdf.name})")
                continue

            # --- SANS prétraitement ---
            print(f"  [OCR SANS | {noise:6s}] en cours...", end="", flush=True)
            txt_sans = run_sans(noisy_pdf)
            m_sans = calc_all(gt_text, txt_sans)
            (DUMP_DIR / f"{cv_stem}_{noise}_SANS_OPENCV.txt").write_text(txt_sans, encoding="utf-8")
            print(f"\r  SANS OpenCV    | {noise:6s}           "
                  f"  {m_sans['CER']:>7.3f} {m_sans['WER']:>7.3f}"
                  f"  {m_sans['Jaccard']:>8.3f} {m_sans['BLEU-4']:>8.4f}")

            # --- AVEC prétraitement v2 ---
            print(f"  [OCR AVEC | {noise:6s}] en cours...", end="", flush=True)
            txt_avec = run_avec(noisy_pdf, cv_stem)
            m_avec = calc_all(gt_text, txt_avec)
            (DUMP_DIR / f"{cv_stem}_{noise}_AVEC_OPENCV.txt").write_text(txt_avec, encoding="utf-8")
            print(f"\r  AVEC OpenCV v2 | {noise:6s}           "
                  f"  {m_avec['CER']:>7.3f} {m_avec['WER']:>7.3f}"
                  f"  {m_avec['Jaccard']:>8.3f} {m_avec['BLEU-4']:>8.4f}")

            # Gain
            cer_gain   = m_sans['CER']    - m_avec['CER']
            jac_gain   = m_avec['Jaccard'] - m_sans['Jaccard']
            bleu_gain  = m_avec['BLEU-4'] - m_sans['BLEU-4']
            print(f"  {'Δ (AVEC−SANS)':<35}  {-cer_gain:>+7.3f} {'':>7}  {jac_gain:>+8.3f} {bleu_gain:>+8.4f}  "
                  f"{'✅' if cer_gain > 0 else '❌'}")

            results.append({
                "cv": cv_stem, "noise": noise,
                **{f"sans_{k}": v for k, v in m_sans.items()},
                **{f"avec_{k}": v for k, v in m_avec.items()},
            })

    # ── Bilan global ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  BILAN GLOBAL PAR NIVEAU DE BRUIT")
    print(f"{'='*72}")
    for noise in NOISE_LEVELS:
        rows = [r for r in results if r["noise"] == noise]
        if not rows: continue
        n = len(rows)
        avg = lambda key: sum(r[key] for r in rows) / n
        print(f"\n  [{noise.upper()}]  (n={n} CVs)")
        for metric in ["CER", "WER", "Jaccard", "BLEU-4"]:
            s = avg(f"sans_{metric}")
            a = avg(f"avec_{metric}")
            sign = "↓" if metric in ("CER","WER") else "↑"
            better = (a < s) if metric in ("CER","WER") else (a > s)
            flag = "✅" if better else "❌"
            print(f"    {metric:8s}  SANS={s:.3f}  AVEC={a:.3f}  Δ={a-s:+.3f} {sign}  {flag}")

    print(f"\n  Dumps mis à jour dans : {DUMP_DIR.resolve()}")
    print("=" * 72)

if __name__ == "__main__":
    run()
