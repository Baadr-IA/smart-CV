import cv2
import numpy as np
from PIL import Image
import fitz
from outils.image_processor import preprocess_for_ocr, get_text_blocks
from pathlib import Path

def test_segmentation_visual():
    pdf_path = Path("input/cv-developpeur-web.pdf")
    if not pdf_path.exists():
        print(f"Erreur : {pdf_path} introuvable.")
        return

    doc = fitz.open(str(pdf_path))
    page = doc[0] # Page 1
    
    # 1. Rendu du PDF en image (300 DPI)
    print("Rendu du PDF en image...")
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 2. Prétraitement (Nettoyage OpenCV)
    print("Nettoyage de l'image (OpenCV)...")
    clean_img = preprocess_for_ocr(img_pil)
    
    # 3. Détection des blocs
    print("Détection des blocs morphologiques...")
    blocks = get_text_blocks(clean_img)
    
    # 4. Dessiner les résultats pour vérification visuelle
    img_cv = cv2.cvtColor(np.array(clean_img), cv2.COLOR_GRAY2BGR)
    
    print(f"\nRésultats : {len(blocks)} blocs détectés.")
    for i, (x, y, w, h) in enumerate(blocks):
        # On dessine un rectangle vert
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # On numérote le bloc pour voir l'ordre de lecture
        cv2.putText(img_cv, f"BLOC {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        print(f"  - Bloc {i} : x={x}, y={y}, w={w}, h={h}")

    # 5. Sauvegarde
    output_path = "debug_segmentation_test.png"
    cv2.imwrite(output_path, img_cv)
    print(f"\nVisualisation sauvegardée dans : {output_path}")
    doc.close()

if __name__ == "__main__":
    test_segmentation_visual()
