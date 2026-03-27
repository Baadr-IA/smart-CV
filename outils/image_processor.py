"""
Traitement d'image avancé pour l'OCR (OpenCV).
Nettoie les scans de CV pour maximiser la précision de l'extraction.

Pipeline appliqué UNIQUEMENT pour le mode pytesseract (fallback local).
Pour docling et vision (LLM), aucune binarisation n'est effectuée —
ces modes envoient le PDF directement à leurs moteurs respectifs.
"""

import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_for_ocr(image_pil: Image.Image) -> Image.Image:
    """
    Pipeline de nettoyage pour maximiser la précision pytesseract :

    0. Upscaling  → garantit ≥ 2400px de hauteur (300 DPI effectif)
    1. Grayscale  → supprime la couleur inutile pour l'OCR
    2. Deskew     → corrige l'inclinaison du scan
    3. Denoising  → fastNlMeansDenoising (meilleur que medianBlur pour le texte)
    4. CLAHE      → boost de contraste local (fonds colorés, CV design)
    5. Binarisation intelligente :
         - Si le scan est propre (fond blanc/texte noir) → Otsu global
           (plus net, moins d'artefacts que l'adaptatif)
         - Si fond complexe / inégal → Adaptive Gaussian (bloc 21, C=10)
           (paramètres calibrés pour texte 10-12pt à 300 DPI)
    6. Morphologie → dilatation légère pour reconnecter les lettres brisées
    """
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # 0. Upscaling — cible 2400px de hauteur pour un OCR fiable à 300 DPI
    h, w = img.shape[:2]
    if h < 2400:
        scale = 2400 / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        logger.debug("Upscaling ×%.2f → %dx%d", scale, img.shape[1], img.shape[0])

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Deskew (correction d'inclinaison via moments)
    gray = _deskew(gray)

    # 3. Denoising — fastNlMeans est plus efficace que medianBlur pour le texte imprimé
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. CLAHE — boost de contraste local pour fonds colorés / CV design
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Binarisation intelligente
    # Détecte si le fond est globalement uniforme (scan propre) ou complexe
    std_dev = float(np.std(enhanced))
    if std_dev < 40:
        # Fond uniforme → Otsu global : résultat plus net, sans effet de mosaïque
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.debug("Binarisation Otsu (std=%.1f)", std_dev)
    else:
        # Fond complexe / irrégulier → Adaptive Gaussian
        # Bloc 21 (vs 11 avant) : fenêtre plus large = meilleure gestion des variations
        # Constante 10 (vs 2 avant) : soustraction plus forte = texte plus net
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,   # block size — doit être impair, ≥ 3
            10,   # constante soustraite
        )
        logger.debug("Binarisation Adaptive Gaussian (std=%.1f)", std_dev)

    # 6. Morphologie légère — reconnecte les lettres brisées par la binarisation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(clean)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Corrige l'inclinaison d'un scan via les moments de l'image binaire.
    Angle détecté par minAreaRect sur les pixels sombres (texte).
    Correction appliquée uniquement si l'angle est significatif (> 0.3°).
    """
    # Binarisation rapide pour détecter l'angle seulement
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 100:
        return gray  # Pas assez de texte pour mesurer l'angle

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect retourne un angle entre -90 et 0 ; on ramène à ±45°
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.3:
        return gray  # Inclinaison négligeable

    logger.debug("Deskew : correction de %.2f°", angle)
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated

