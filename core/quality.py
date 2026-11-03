"""
Retinal Image Quality Assessment Module for NetraSathi.
Checks:
1. Blur detection via Variance of Laplacian
2. Luminance & exposure uniformity
3. Retinal circular field of view coverage
Outputs an aggregated quality score (0.0 - 1.0) and helpful warnings.
"""
import cv2
import numpy as np

def assess_image_quality(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    circle_info: tuple[int, int, int]
) -> dict:
    """
    Evaluates retinal image quality.
    Returns:
    {
        'is_acceptable': bool,
        'quality_score': float, # 0.0 to 1.0
        'blur_score': float,   # Laplacian variance
        'brightness_score': float, # Mean luminance inside mask (0-255)
        'coverage_ratio': float, # Ratio of retinal circle to total area
        'warnings': list[str],
        'details': dict
    }
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur Assessment: Variance of Laplacian within retinal mask
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Only calculate inside mask
    retina_laplacian = laplacian[mask > 0]
    blur_var = float(np.var(retina_laplacian)) if len(retina_laplacian) > 0 else 0.0
    
    # Typical retinal thresholds:
    # < 35: Severely blurry
    # 35 - 75: Marginal
    # > 75: Crisp / Good details
    blur_norm = min(1.0, max(0.0, (blur_var - 20) / 100.0))
    
    # 2. Exposure / Brightness Assessment inside retinal mask
    retina_pixels = gray[mask > 0]
    mean_brightness = float(np.mean(retina_pixels)) if len(retina_pixels) > 0 else 0.0
    
    # Ideal retinal mean brightness is ~50 to 170
    if mean_brightness < 40:
        brightness_norm = max(0.0, mean_brightness / 40.0)
    elif mean_brightness > 210:
        brightness_norm = max(0.0, (255 - mean_brightness) / 45.0)
    else:
        brightness_norm = 1.0
        
    # 3. Retinal Coverage Ratio
    total_pixels = h * w
    mask_pixels = np.count_nonzero(mask)
    coverage_ratio = float(mask_pixels / total_pixels)
    coverage_norm = min(1.0, max(0.0, coverage_ratio / 0.40))
    
    # Aggregate weighted score
    quality_score = float(0.50 * blur_norm + 0.35 * brightness_norm + 0.15 * coverage_norm)
    quality_score = round(min(1.0, max(0.0, quality_score)), 2)
    
    warnings = []
    if blur_var < 45:
        warnings.append("Retake photo: image is blurry. Retinal vessels lack sharp focus.")
    if mean_brightness < 40:
        warnings.append("Retake photo: image is too dark (underexposed).")
    elif mean_brightness > 210:
        warnings.append("Retake photo: glare or overexposure detected.")
    if coverage_ratio < 0.25:
        warnings.append("Retina occupies too little of the camera frame. Center the eye closer.")
        
    is_acceptable = (quality_score >= 0.50 and blur_var >= 40)
    
    return {
        "is_acceptable": is_acceptable,
        "quality_score": quality_score,
        "blur_score": round(blur_var, 1),
        "brightness_score": round(mean_brightness, 1),
        "coverage_ratio": round(coverage_ratio, 2),
        "warnings": warnings,
        "details": {
            "blur_normalized": round(blur_norm, 2),
            "brightness_normalized": round(brightness_norm, 2),
            "coverage_normalized": round(coverage_norm, 2)
        }
    }
