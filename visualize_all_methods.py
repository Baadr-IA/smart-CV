
import cv2
import numpy as np
from PIL import Image
import fitz
from pathlib import Path
from outils.image_processor import preprocess_for_ocr, get_text_blocks
from outils.parser import _parse_via_docling

def visualize_opencv(pdf_path):
    doc = fitz.open(str(pdf_path))
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    clean_img = preprocess_for_ocr(img_pil)
    blocks = get_text_blocks(clean_img)
    
    img_cv = cv2.cvtColor(np.array(clean_img), cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(blocks):
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_cv, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite("view_opencv_morphology.png", img_cv)
    print("-> view_opencv_morphology.png créé.")
    doc.close()

def visualize_pymupdf(pdf_path):
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img_cv = cv2.cvtColor(np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples)), cv2.COLOR_RGB2BGR)
    
    # Récupérer les blocs natifs du PDF
    blocks = page.get_text("blocks")
    for i, b in enumerate(blocks):
        # b = (x0, y0, x1, y1, "texte", block_no, block_type)
        # On doit scaler les coordonnées (72 DPI -> 300 DPI)
        scale = 300/72
        x0, y0, x1, y1 = int(b[0]*scale), int(b[1]*scale), int(b[2]*scale), int(b[3]*scale)
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), (255, 0, 0), 3)
        cv2.putText(img_cv, str(i), (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    cv2.imwrite("view_pymupdf_blocks.png", img_cv)
    print("-> view_pymupdf_blocks.png créé.")
    doc.close()

def visualize_docling(pdf_path):
    # Note: Docling ne donne pas facilement les bounding boxes sans re-calculer
    # On va simuler sa vue en utilisant Docling pour extraire et logger sa structure
    print("Extraction via Docling (ceci peut prendre 30s)...")
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        # Docling exporte en Markdown, on ne peut pas dessiner facilement 
        # mais on peut dire que c'est sa sortie de référence.
        with open("view_docling_output.md", "w", encoding="utf-8") as f:
            f.write(result.document.export_to_markdown())
        print("-> view_docling_output.md créé (Structure IA).")
    except Exception as e:
        print(f"Erreur Docling : {e}")

if __name__ == "__main__":
    p = Path("input/cv-developpeur-web.pdf")
    visualize_opencv(p)
    visualize_pymupdf(p)
    visualize_docling(p)
