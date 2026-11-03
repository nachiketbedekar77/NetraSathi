"""
End-to-end test script for NetraSathi core modules.
"""
import os
import cv2
import numpy as np
from PIL import Image

from core.db import init_db, add_patient, get_patient, add_scan, get_patient_scans
from core.preprocess import preprocess_retinal_image
from core.quality import assess_image_quality
from core.model import predict, GRADE_NAMES
from core.gradcam import generate_gradcam_overlay
from core.report import generate_pdf_report

def run_tests():
    print("1. Testing Database...")
    init_db()
    pid = add_patient(
        name="Ramesh Patil",
        age=58,
        sex="Male",
        phone="9876543210",
        village="Shirwal",
        years_diabetes=7.5
    )
    patient = get_patient(pid)
    assert patient["name"] == "Ramesh Patil"
    print(f"Patient registered with ID: {pid}")

    print("\n2. Testing Sample Loading and Preprocessing...")
    sample_path = "samples/sample_moderate.jpg"
    assert os.path.exists(sample_path), "Sample file missing!"
    img_bgr = cv2.imread(sample_path)
    raw_bgr, cleaned_bgr, mask, circle_info = preprocess_retinal_image(img_bgr)
    assert cleaned_bgr.shape == (512, 512, 3)
    assert mask.shape == (512, 512)
    print("Preprocessing successful! Mask radius:", circle_info[2])

    print("\n3. Testing Quality Assessment...")
    quality = assess_image_quality(raw_bgr, mask, circle_info)
    print(f"Quality Score: {quality['quality_score']}, Blur Var: {quality['blur_score']}, Acceptable: {quality['is_acceptable']}")

    print("\n4. Testing Prediction...")
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    grade, probs, latency, is_demo = predict(raw_rgb)
    print(f"Predicted Grade: {grade} ({GRADE_NAMES[grade]}), Latency: {latency}s, Demo: {is_demo}")
    print(f"Probabilities: {[round(p, 3) for p in probs]}")

    print("\n5. Testing Grad-CAM Explainability...")
    overlay_rgb, heatmap = generate_gradcam_overlay(raw_rgb, mask, target_class=grade, alpha=0.40)
    assert overlay_rgb.shape == (512, 512, 3)
    assert heatmap.shape == (512, 512)
    # Check that heatmap outside mask is exactly 0
    assert np.all(heatmap[mask == 0] == 0.0)
    print("Grad-CAM generation successful! Heatmap strictly within retinal mask.")

    print("\n6. Testing PDF Generation...")
    scan_meta = {
        "eye": "Left Eye (OS)",
        "grade": grade,
        "grade_name": GRADE_NAMES[grade],
        "confidence": probs[grade],
        "quality_score": quality["quality_score"],
        "blur_score": quality["blur_score"],
        "referral_advice": "Refer to eye specialist within 3 months"
    }
    orig_pil = Image.fromarray(raw_rgb)
    overlay_pil = Image.fromarray(overlay_rgb)
    pdf_bytes = generate_pdf_report(patient, scan_meta, orig_pil, overlay_pil, is_demo=is_demo)
    assert len(pdf_bytes) > 5000, "PDF bytes too small!"
    os.makedirs("data", exist_ok=True)
    with open("data/test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF generated successfully ({len(pdf_bytes)} bytes) -> data/test_report.pdf")

    print("\n7. Recording scan to database...")
    scan_id = add_scan(
        patient_id=pid,
        eye="Left Eye (OS)",
        grade=grade,
        grade_name=GRADE_NAMES[grade],
        confidence=probs[grade],
        quality_score=quality["quality_score"],
        blur_score=quality["blur_score"],
        brightness_score=quality["brightness_score"],
        image_path="samples/sample_moderate.jpg",
        heatmap_path="data/test_heatmap.jpg",
        referral_advice=scan_meta["referral_advice"]
    )
    scans = get_patient_scans(pid)
    assert len(scans) >= 1
    print(f"Scan recorded with ID: {scan_id}. Total scans for patient: {len(scans)}")

    print("\nALL CORE BACKEND MODULE TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
