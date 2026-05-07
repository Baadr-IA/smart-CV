
import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def preprocess_for_ocr(image_pil: Image.Image, deskew: bool = True) -> Image.Image:
    """
    Pipeline OCR-aware :
    - Toujours redimensionné à ~2000px hauteur en sortie (Tesseract optimal + évite timeout >6000px)
    - Images propres (std_dev < 35) : sharpening léger + Otsu
    - Images bruitées modérées (35-60) : NL-Means h=8 + Otsu
    - Images fortement bruitées (> 60) : NL-Means h=15 + AdaptiveThreshold
    - NLM appliqué à résolution réduite (≤ 1200px H) pour la performance
    - deskew=False recommandé pour les CV multi-colonnes
    """
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Diagnostic du niveau de bruit sur l'image originale
    std_dev = float(np.std(gray))

    # 2. Débruitage NLM à résolution réduite (performance : NLM = O(N²) sur grande image)
    if std_dev > 35:
        h_orig, w_orig = gray.shape[:2]
        if h_orig > 1200:
            scale_down = 1200 / h_orig
            gray_small = cv2.resize(gray, None, fx=scale_down, fy=scale_down,
                                    interpolation=cv2.INTER_AREA)
        else:
            gray_small = gray
            scale_down = 1.0

        h_nlm = 15 if std_dev > 60 else 8
        denoised_small = cv2.fastNlMeansDenoising(
            gray_small, h=h_nlm, templateWindowSize=7, searchWindowSize=11)

        if scale_down < 1.0:
            gray = cv2.resize(denoised_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        else:
            gray = denoised_small

    # 3. Normaliser à 2000px hauteur (Tesseract échoue silencieusement sur > ~6000px)
    h, w = gray.shape[:2]
    target_h = 2000
    if h != target_h:
        scale = target_h / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # 4. Deskew (désactivable pour les layouts multi-colonnes)
    if deskew:
        gray = _deskew(gray)

    # 5. Binarisation
    if std_dev > 60:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )
    elif std_dev > 35:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(binary)

def get_text_blocks(image_pil: Image.Image):
    """
    Segmentation par blocs optimisée.
    """
    img = np.array(image_pil)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Inversion pour dilatation
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphologie avec kernel rectangulaire
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 10:
            blocks.append((x, y, w, h))

    # Split colonnes rapide
    split_x = get_column_split_x(image_pil)
    if split_x:
        blocks.sort(key=lambda b: (0 if (b[0] + b[2]/2) < split_x else 1, b[1]))
    else:
        blocks.sort(key=lambda b: (b[0] // 500, b[1])) 
    
    return blocks

def get_column_split_x(image_pil: Image.Image):
    # On travaille sur une version réduite pour aller très vite
    small = np.array(image_pil.convert('L').resize((500, 700)))
    _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = np.sum(thresh, axis=0)
    
    width = len(projection)
    search_start, search_end = int(width * 0.25), int(width * 0.45)
    
    if search_start < search_end:
        split_x_small = search_start + np.argmin(projection[search_start:search_end])
        # On remet à l'échelle d'origine
        return int(split_x_small * (np.array(image_pil).shape[1] / 500))
    return None

def _deskew(gray: np.ndarray) -> np.ndarray:
    # Calcul d'angle sur image réduite pour la vitesse
    small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
    _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100: return gray
    
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = 90 + angle
    if abs(angle) < 0.5: return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
