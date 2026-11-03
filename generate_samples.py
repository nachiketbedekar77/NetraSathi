"""
Generate synthetic fundus-style retinal images for NetraSathi college demo.
Creates:
1. samples/sample_normal.jpg (Grade 0: No DR)
2. samples/sample_moderate.jpg (Grade 2: Moderate NPDR)
3. samples/sample_severe.jpg (Grade 4: Proliferative DR)
"""
import os
import math
import random
import numpy as np
import cv2

def create_base_fundus(width=512, height=512, eye="right"):
    # Create black canvas
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    radius = int(min(width, height) * 0.44)

    # 1. Circular gradient background for retina (deep orange-red to darker edge)
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist <= radius

    # Radial color variation
    norm_dist = np.clip(dist / radius, 0, 1)
    
    # Base retinal color (BGR)
    # Center: bright orange-red (B: 15, G: 65, R: 185)
    # Edge: dark brick red (B: 5, G: 20, R: 85)
    b = (15 * (1 - norm_dist**1.5) + 5 * (norm_dist**1.5)).astype(np.uint8)
    g = (65 * (1 - norm_dist**1.3) + 22 * (norm_dist**1.3)).astype(np.uint8)
    r = (195 * (1 - norm_dist**1.2) + 95 * (norm_dist**1.2)).astype(np.uint8)
    
    fundus = np.dstack((b, g, r))
    img[mask] = fundus[mask]

    # Smooth the retinal edge
    edge_mask = (dist > radius - 6) & (dist <= radius)
    alpha = (radius - dist[edge_mask]) / 6.0
    for c in range(3):
        img[edge_mask, c] = (img[edge_mask, c] * alpha).astype(np.uint8)

    # 2. Optic Disc (yellow-pink oval)
    # In right eye: optic disc is nasal (left side of image ~35% cx)
    # In left eye: optic disc is nasal (right side ~65% cx)
    disc_x = int(cx - radius * 0.45) if eye == "right" else int(cx + radius * 0.45)
    disc_y = cy
    cv2.ellipse(img, (disc_x, disc_y), (int(radius * 0.17), int(radius * 0.22)), 0, 0, 360, (110, 190, 245), -1)
    cv2.ellipse(img, (disc_x, disc_y), (int(radius * 0.11), int(radius * 0.14)), 0, 0, 360, (140, 215, 255), -1)

    # 3. Macula / Fovea (darker avascular zone temporal to optic disc)
    fovea_x = int(cx + radius * 0.3) if eye == "right" else int(cx - radius * 0.3)
    fovea_y = cy + int(radius * 0.05)
    cv2.circle(img, (fovea_x, fovea_y), int(radius * 0.16), (10, 35, 110), -1)
    cv2.circle(img, (fovea_x, fovea_y), int(radius * 0.05), (5, 20, 75), -1)

    # 4. Retinal Blood Vessels (branching tree from optic disc)
    # Superior & Inferior arcades
    angles = [math.radians(-65), math.radians(-35), math.radians(35), math.radians(65), math.radians(180)]
    for base_ang in angles:
        curr_x, curr_y = float(disc_x), float(disc_y)
        step_len = 6.0
        ang = base_ang
        thickness = 4
        for step in range(40):
            if step > 10: thickness = 3
            if step > 20: thickness = 2
            if step > 30: thickness = 1

            next_x = curr_x + step_len * math.cos(ang)
            next_y = curr_y + step_len * math.sin(ang)

            # Keep inside fundus circle
            if math.hypot(next_x - cx, next_y - cy) > radius - 10:
                break

            cv2.line(img, (int(curr_x), int(curr_y)), (int(next_x), int(next_y)), (15, 15, 80), thickness)
            
            # Add occasional branching
            if step == 15:
                branch_ang = ang + 0.4
                bx, by = next_x, next_y
                for bstep in range(15):
                    nbx = bx + step_len * 0.8 * math.cos(branch_ang)
                    nby = by + step_len * 0.8 * math.sin(branch_ang)
                    if math.hypot(nbx - cx, nby - cy) > radius - 10: break
                    cv2.line(img, (int(bx), int(by)), (int(nbx), int(nby)), (15, 15, 85), 1)
                    bx, by = nbx, nby

            curr_x, curr_y = next_x, next_y
            ang += random.uniform(-0.08, 0.08)

    # Subtle optical blur to look organic
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img, (cx, cy, radius)

def add_moderate_dr_lesions(img, cx, cy, radius):
    random.seed(42)
    # Microaneurysms (pinpoint red dots, BGR: 5, 5, 120)
    for _ in range(35):
        r_dist = random.uniform(radius * 0.15, radius * 0.75)
        angle = random.uniform(0, 2 * math.pi)
        lx = int(cx + r_dist * math.cos(angle))
        ly = int(cy + r_dist * math.sin(angle))
        cv2.circle(img, (lx, ly), random.randint(1, 3), (10, 10, 130), -1)

    # Hard Exudates (bright yellowish waxy deposits, BGR: 90, 210, 240)
    exudate_cluster_x = cx + int(radius * 0.25)
    exudate_cluster_y = cy - int(radius * 0.2)
    for _ in range(25):
        lx = int(exudate_cluster_x + random.gauss(0, radius * 0.08))
        ly = int(exudate_cluster_y + random.gauss(0, radius * 0.08))
        if math.hypot(lx - cx, ly - cy) < radius - 15:
            cv2.ellipse(img, (lx, ly), (random.randint(2, 5), random.randint(2, 4)), random.randint(0, 180), 0, 360, (110, 220, 250), -1)

    # Blot hemorrhages (dark red irregular spots)
    for _ in range(8):
        r_dist = random.uniform(radius * 0.25, radius * 0.65)
        angle = random.uniform(0, 2 * math.pi)
        lx = int(cx + r_dist * math.cos(angle))
        ly = int(cy + r_dist * math.sin(angle))
        cv2.ellipse(img, (lx, ly), (random.randint(5, 9), random.randint(4, 7)), random.randint(0, 180), 0, 360, (10, 15, 95), -1)

    return img

def add_severe_dr_lesions(img, cx, cy, radius):
    random.seed(99)
    # First apply moderate lesions
    img = add_moderate_dr_lesions(img, cx, cy, radius)

    # Cotton Wool Spots (soft, fluffy whitish patches, nerve fiber infarcts)
    cws_spots = [
        (cx - int(radius * 0.15), cy + int(radius * 0.3)),
        (cx + int(radius * 0.4), cy + int(radius * 0.15)),
        (cx - int(radius * 0.2), cy - int(radius * 0.35)),
    ]
    for px, py in cws_spots:
        patch = np.zeros_like(img)
        cv2.circle(patch, (px, py), int(radius * 0.09), (180, 210, 230), -1)
        patch = cv2.GaussianBlur(patch, (21, 21), 10)
        img = cv2.addWeighted(img, 1.0, patch, 0.6, 0)

    # Large flame hemorrhages & extensive blot hemorrhages
    for _ in range(16):
        r_dist = random.uniform(radius * 0.15, radius * 0.8)
        angle = random.uniform(0, 2 * math.pi)
        lx = int(cx + r_dist * math.cos(angle))
        ly = int(cy + r_dist * math.sin(angle))
        cv2.ellipse(img, (lx, ly), (random.randint(7, 16), random.randint(4, 9)), random.randint(0, 180), 0, 360, (5, 5, 80), -1)

    # Neovascularization (frond-like fine tangled abnormal vessels near disc and retina)
    for _ in range(20):
        sx = cx - int(radius * 0.35) + random.randint(-20, 20)
        sy = cy + random.randint(-30, 30)
        ex = sx + random.randint(-18, 18)
        ey = sy + random.randint(-18, 18)
        cv2.line(img, (sx, sy), (ex, ey), (15, 20, 95), 1)

    return img

def generate_all_samples():
    os.makedirs("samples", exist_ok=True)

    # 1. Normal (Grade 0: No DR)
    img_normal, (cx, cy, rad) = create_base_fundus(512, 512, eye="right")
    cv2.imwrite("samples/sample_normal.jpg", img_normal, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("Generated samples/sample_normal.jpg (Grade 0: No DR)")

    # 2. Moderate NPDR (Grade 2)
    img_mod, (cx, cy, rad) = create_base_fundus(512, 512, eye="left")
    img_mod = add_moderate_dr_lesions(img_mod, cx, cy, rad)
    cv2.imwrite("samples/sample_moderate.jpg", img_mod, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("Generated samples/sample_moderate.jpg (Grade 2: Moderate NPDR)")

    # 3. Severe / Proliferative DR (Grade 4)
    img_sev, (cx, cy, rad) = create_base_fundus(512, 512, eye="right")
    img_sev = add_severe_dr_lesions(img_sev, cx, cy, rad)
    cv2.imwrite("samples/sample_severe.jpg", img_sev, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("Generated samples/sample_severe.jpg (Grade 4: Proliferative DR)")

if __name__ == "__main__":
    generate_all_samples()

