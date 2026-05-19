"""
Benchmark OCR for CV PDFs using project-native extractors:
- pypdf (native text)
- docling (OCR + layout)
- pytesseract (OCR with preprocessing)

Usage examples:
  python bench_ocr.py --input-dir "C:\\path\\to\\pdfs" --limit 50
  python bench_ocr.py --input-dir "C:\\path\\to\\pdfs" --methods pypdf docling tesseract --limit 100
  python bench_ocr.py --hf-dataset "lhoestq/resumes-raw-pdf-for-ocr" --hf-limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

from outils.parser import (
    _parse_pdf_native,
    _parse_via_docling,
    _ocr_pdf,
    _clean_text,
    _compute_text_quality,
    _is_quality_sufficient,
    OCR_MIN_ALPHA_RATIO,
    OCR_MIN_CHARS,
)


try:
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter
except Exception:
    Image = None
    ImageChops = None
    ImageEnhance = None
    ImageFilter = None


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _cer(reference: str, hypothesis: str) -> Optional[float]:
    if reference is None:
        return None
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return None
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def _wer(reference: str, hypothesis: str) -> Optional[float]:
    if reference is None:
        return None
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return None
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def _iter_pdfs(paths: Iterable[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path)
        elif path.is_dir():
            pdfs.extend(sorted(path.rglob("*.pdf")))
    return pdfs


def _load_hf_dataset(dataset_name: str, split: str, limit: Optional[int]) -> list[Path]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets n'est pas installé. Installe-le avec `pip install datasets` "
            "ou utilise --input-dir."
        ) from exc

    ds = load_dataset(dataset_name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    temp_dir = Path(tempfile.mkdtemp(prefix="hf_resumes_"))
    pdf_paths: list[Path] = []
    for idx, row in enumerate(ds):
        pdf_obj = row.get("pdf") or row.get("file") or row.get("document")
        if pdf_obj is None:
            continue

        if isinstance(pdf_obj, dict) and "bytes" in pdf_obj:
            data = pdf_obj["bytes"]
        elif isinstance(pdf_obj, bytes):
            data = pdf_obj
        elif isinstance(pdf_obj, str) and os.path.exists(pdf_obj):
            pdf_paths.append(Path(pdf_obj))
            continue
        else:
            continue

        out_path = temp_dir / f"resume_{idx:05d}.pdf"
        out_path.write_bytes(data)
        pdf_paths.append(out_path)

    return pdf_paths


def _load_gt_from_zip(zip_path: Path) -> dict[str, str]:
    import zipfile

    gt_map: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            stem = Path(name).stem.replace("_annotated", "")
            raw = zf.read(name).decode("utf-8", errors="ignore")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = data.get("text", "")
            if text:
                gt_map[stem] = _clean_text(text)
    return gt_map


def _render_pdf_pages(pdf_path: Path) -> list[Image.Image]:
    if Image is None:
        raise RuntimeError("Pillow requis pour le rendu des pages PDF.")
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PyMuPDF requis pour le rendu des PDFs.") from exc

    doc = fitz.open(str(pdf_path))
    images: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def _apply_noise(img: Image.Image, level: str) -> Image.Image:
    if level == "none":
        return img

    if ImageEnhance is None or ImageFilter is None or ImageChops is None:
        return img

    if level == "light":
        contrast = 0.9
        blur_radius = 0.5
        noise_sigma = 8
        rotate = 0.5
    elif level == "medium":
        contrast = 0.8
        blur_radius = 1.0
        noise_sigma = 15
        rotate = 1.0
    else:
        contrast = 0.7
        blur_radius = 1.5
        noise_sigma = 25
        rotate = 1.5

    base = img.convert("L")
    base = ImageEnhance.Contrast(base).enhance(contrast)
    if blur_radius:
        base = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    noise = Image.effect_noise(base.size, noise_sigma)
    noisy = ImageChops.add(base, noise, scale=2.0)

    if rotate:
        noisy = noisy.rotate(rotate, expand=True, fillcolor=255)
    return noisy.convert("RGB")


def _images_to_pdf(images: list[Image.Image], output_path: Path) -> None:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PyMuPDF requis pour la création de PDF.") from exc

    doc = fitz.open()
    for img in images:
        width, height = img.size
        page = doc.new_page(width=width, height=height)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name, format="PNG")
            page.insert_image(page.rect, filename=tmp.name)
        os.unlink(tmp.name)
    doc.save(str(output_path))
    doc.close()


def _prepare_noisy_pdf(original: Path, noise_level: str, cache_dir: Path) -> Path:
    if noise_level == "none":
        return original
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{original.stem}__{noise_level}.pdf"
    if out_path.exists():
        return out_path
    pages = _render_pdf_pages(original)
    noisy_pages = [_apply_noise(img, noise_level) for img in pages]
    _images_to_pdf(noisy_pages, out_path)
    return out_path


def _run_method(file_path: Path, method: str, gt_text: Optional[str], noise_level: str) -> dict:
    start = time.perf_counter()
    text = ""
    used_method = method
    is_markdown = False
    error = None

    try:
        if method == "pypdf":
            text, used_method = _parse_pdf_native(file_path)
        elif method == "docling":
            text = _parse_via_docling(file_path)
            used_method = "docling-markdown"
            is_markdown = True
        elif method == "tesseract":
            text = _ocr_pdf(file_path)
            used_method = "pytesseract-ocr"
        else:
            raise ValueError(f"Méthode inconnue: {method}")
    except Exception as exc:
        error = str(exc)

    duration = time.perf_counter() - start
    cleaned = _clean_text(text, is_markdown=is_markdown) if text else ""
    quality = _compute_text_quality(cleaned)
    quality["sufficient"] = _is_quality_sufficient(quality)
    cer_value = _cer(gt_text, cleaned) if gt_text else None
    wer_value = _wer(gt_text, cleaned) if gt_text else None

    return {
        "file": str(file_path),
        "method": used_method,
        "noise_level": noise_level,
        "duration_sec": round(duration, 4),
        "char_count": quality["char_count"],
        "word_count": quality["word_count"],
        "line_count": quality["line_count"],
        "alpha_ratio": quality["alpha_ratio"],
        "sufficient": quality["sufficient"],
        "cer": round(cer_value, 4) if cer_value is not None else None,
        "wer": round(wer_value, 4) if wer_value is not None else None,
        "error": error,
    }


def _summarize(records: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for method in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == method]
        if not rows:
            continue
        durations = [r["duration_sec"] for r in rows if r["error"] is None]
        chars = [r["char_count"] for r in rows if r["error"] is None]
        alpha = [r["alpha_ratio"] for r in rows if r["error"] is None]
        cer_values = [r["cer"] for r in rows if r["cer"] is not None and r["error"] is None]
        wer_values = [r["wer"] for r in rows if r["wer"] is not None and r["error"] is None]
        success = [r for r in rows if r["error"] is None]
        sufficient = [r for r in rows if r["sufficient"] and r["error"] is None]
        summary[method] = {
            "files": len(rows),
            "ok": len(success),
            "errors": len(rows) - len(success),
            "sufficient": len(sufficient),
            "min_chars_threshold": OCR_MIN_CHARS,
            "min_alpha_ratio": OCR_MIN_ALPHA_RATIO,
            "avg_duration_sec": round(sum(durations) / max(len(durations), 1), 4),
            "avg_char_count": round(sum(chars) / max(len(chars), 1), 2),
            "avg_alpha_ratio": round(sum(alpha) / max(len(alpha), 1), 4),
            "avg_cer": round(sum(cer_values) / max(len(cer_values), 1), 4) if cer_values else None,
            "avg_wer": round(sum(wer_values) / max(len(wer_values), 1), 4) if wer_values else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OCR pipelines for CV PDFs.")
    parser.add_argument("--input-dir", action="append", help="Directory with PDF files (can be repeated).")
    parser.add_argument("--hf-dataset", help="HuggingFace dataset name (ex: lhoestq/resumes-raw-pdf-for-ocr).")
    parser.add_argument("--hf-split", default="train", help="Split name for HF dataset.")
    parser.add_argument("--hf-limit", type=int, default=None, help="Max HF samples.")
    parser.add_argument("--limit", type=int, default=None, help="Max total PDFs to process.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--methods", nargs="+", default=["pypdf", "docling", "tesseract"])
    parser.add_argument("--output", default="bench_output", help="Output directory for results.")
    parser.add_argument("--gt-zip", help="ZIP containing JSON ground truth with 'text' field.")
    parser.add_argument("--noise-levels", nargs="+", default=["none"], help="Noise levels: none light medium heavy")
    args = parser.parse_args()

    sources: list[Path] = []
    if args.input_dir:
        sources.extend(Path(p) for p in args.input_dir)

    pdfs: list[Path] = []
    if sources:
        pdfs.extend(_iter_pdfs(sources))

    if args.hf_dataset:
        pdfs.extend(_load_hf_dataset(args.hf_dataset, args.hf_split, args.hf_limit))

    if not pdfs:
        raise SystemExit("Aucun PDF trouvé. Utilise --input-dir et/ou --hf-dataset.")

    gt_map: dict[str, str] = {}
    if args.gt_zip:
        gt_map = _load_gt_from_zip(Path(args.gt_zip))

    rng = random.Random(args.seed)
    rng.shuffle(pdfs)
    if args.limit:
        pdfs = pdfs[: args.limit]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for idx, pdf in enumerate(pdfs, start=1):
        print(f"[{idx}/{len(pdfs)}] {pdf.name}")
        stem = pdf.stem
        gt_text = gt_map.get(stem)
        cache_dir = output_dir / "noisy_pdfs"
        for noise_level in args.noise_levels:
            working_pdf = _prepare_noisy_pdf(pdf, noise_level, cache_dir)
            for method in args.methods:
                record = _run_method(working_pdf, method, gt_text, noise_level)
                records.append(record)
                print(
                    f"  - {record['method']} ({noise_level}): "
                    f"{record['char_count']} chars, "
                    f"alpha={record['alpha_ratio']}, "
                    f"{record['duration_sec']}s"
                )

    summary = _summarize(records)

    (output_dir / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
