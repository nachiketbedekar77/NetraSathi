"""
Retinal Image Preprocessing Module for NetraSathi.
Implements:
1. Circular retinal fundus mask extraction
2. Ben Graham's method (local Gaussian blur subtraction + scaling)
3. Green-channel Contrast Limited Adaptive Histogram Equalization (CLAHE)
4. Specular reflection / glare suppression
"""
import cv2
import numpy as np
from PIL import Image

def get_circular_mask(image_bgr: np.ndarray, tolerance: int = 15) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Detect the circular retinal boundary and create a binary mask.
    Returns (mask, (center_x, center_y, radius)).
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Threshold out background
    _, thresh = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    
    # Morphological closing to fill vessels and optic cup holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour representing fundus circle
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius * 0.98) # Slight inward margin
        cx, cy = int(x), int(y)
    else:
        cx, cy = w // 2, h // 2
        radius = int(min(w, h) * 0.48)
        
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask, (cx, cy, radius)

def suppress_glare(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Suppresses bright specular reflections / smartphone flash glare
    inside the fundus area using inpainting on saturated pixels.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect over-saturated white glare spots inside retina
    glare_mask = (gray > 240) & (mask > 0)
    glare_mask = glare_mask.astype(np.uint8) * 255
    
    # Inpaint small glare spots with surrounding fundus texture
    if np.any(glare_mask):
        dilated_glare = cv2.dilate(glare_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cleaned = cv2.inpaint(image_bgr, dilated_glare, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cleaned
    return image_bgr

def ben_graham_enhance(image_bgr: np.ndarray, mask: np.ndarray, sigma: int = 30) -> np.ndarray:
    """
    Ben Graham's award-winning preprocessing method:
    Subtract local Gaussian illumination and scale contrast:
    I_out = alpha * I - beta * Gaussian(I, sigma) + 128
    """
    # 1. Apply Gaussian blur to capture illumination field
    blur = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    
    # 2. Add weighted subtraction: 4 * image - 4 * blur + 128
    enhanced = cv2.addWeighted(image_bgr, 4.0, blur, -4.0, 128)
    
    # 3. Mask out non-retinal background cleanly to black
    enhanced[mask == 0] = 0
    return enhanced

def clahe_green_channel(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE specifically to the green channel (optimal for
    retinal micro-lesions, hemorrhages, and exudates detection) and LAB L-channel.
    """
    # Convert BGR to LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    lab_merged = cv2.merge((l_clahe, a, b))
    bgr_clahe = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    
    # Enhance Green channel directly
    b_ch, g_ch, r_ch = cv2.split(bgr_clahe)
    g_enhanced = clahe.apply(g_ch)
    result = cv2.merge((b_ch, g_enhanced, r_ch))
    
    # Ensure background outside mask stays black
    result[mask == 0] = 0
    return result

def preprocess_retinal_image(image_input) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Full NetraSathi preprocessing pipeline:
    Accepts PIL.Image or numpy BGR array.
    Returns:
    (raw_bgr, cleaned_bgr, mask, (cx, cy, radius))
    """
    if isinstance(image_input, Image.Image):
        raw_rgb = np.array(image_input.convert("RGB"))
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
    else:
        raw_bgr = image_input.copy()
        
    # Resize to standard size for consistent processing (512x512)
    raw_bgr = cv2.resize(raw_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Step 1: Detect circular fundus mask
    mask, circle_info = get_circular_mask(raw_bgr)
    
    # Step 2: Glare suppression
    glare_suppressed = suppress_glare(raw_bgr, mask)
    
    # Step 3: Ben Graham illumination balancing
    ben_graham = ben_graham_enhance(glare_suppressed, mask, sigma=25)
    
    # Step 4: CLAHE enhancement on microvascular features
    cleaned_bgr = clahe_green_channel(ben_graham, mask)
    
    return raw_bgr, cleaned_bgr, mask, circle_info

