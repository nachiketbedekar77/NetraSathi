"""
Generate NetraSathi Logo (PNG and SVG)
Modern medical aesthetic: Teal (#00838F), Cyan (#00ACC1), Deep Blue (#0F172A), Gold accent (#F59E0B)
"""
import os
import math
from PIL import Image, ImageDraw, ImageFont

def generate_svg_logo(output_path):
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="100%">
  <defs>
    <linearGradient id="tealGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00ACC1" />
      <stop offset="50%" stop-color="#00838F" />
      <stop offset="100%" stop-color="#004D40" />
    </linearGradient>
    <linearGradient id="irisGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8" />
      <stop offset="60%" stop-color="#0284C7" />
      <stop offset="100%" stop-color="#0369A1" />
    </linearGradient>
    <linearGradient id="pupilGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A" />
      <stop offset="100%" stop-color="#1E293B" />
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Background soft circle -->
  <circle cx="250" cy="250" r="230" fill="#F0FDFA" stroke="#00838F" stroke-width="4" stroke-opacity="0.2"/>

  <!-- Outer Eye Arc Top & Bottom -->
  <path d="M 60 250 Q 250 80 440 250 Q 250 420 60 250 Z" 
        fill="none" stroke="url(#tealGrad)" stroke-width="18" stroke-linecap="round" stroke-linejoin="round" />

  <!-- Inner Retinal Scan Ring -->
  <circle cx="250" cy="250" r="105" fill="#E0F2FE" stroke="#00ACC1" stroke-width="6" stroke-dasharray="8 6" />

  <!-- Iris -->
  <circle cx="250" cy="250" r="76" fill="url(#irisGrad)" filter="url(#glow)"/>

  <!-- Pupil -->
  <circle cx="250" cy="250" r="38" fill="url(#pupilGrad)"/>

  <!-- Retinal Ray Highlights (Simulating AI Fundus Optics) -->
  <circle cx="236" cy="236" r="12" fill="#FFFFFF" opacity="0.9" />
  <circle cx="264" cy="264" r="6" fill="#FFFFFF" opacity="0.7" />

  <!-- AI Crosshairs / Diagnostic Bracket Overlay -->
  <path d="M 220 185 L 205 185 L 205 200" fill="none" stroke="#F59E0B" stroke-width="5" stroke-linecap="round"/>
  <path d="M 280 185 L 295 185 L 295 200" fill="none" stroke="#F59E0B" stroke-width="5" stroke-linecap="round"/>
  <path d="M 205 300 L 205 315 L 220 315" fill="none" stroke="#F59E0B" stroke-width="5" stroke-linecap="round"/>
  <path d="M 295 300 L 295 315 L 280 315" fill="none" stroke="#F59E0B" stroke-width="5" stroke-linecap="round"/>

  <!-- AI Pulse Wave -->
  <path d="M 120 250 L 160 250 L 175 220 L 190 280 L 205 250 L 250 250" fill="none" stroke="#00838F" stroke-width="4" stroke-linecap="round" opacity="0.6"/>
  <path d="M 250 250 L 295 250 L 310 220 L 325 280 L 340 250 L 380 250" fill="none" stroke="#00838F" stroke-width="4" stroke-linecap="round" opacity="0.6"/>
</svg>"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"SVG logo written to {output_path}")

def generate_png_logo(output_path):
    size = (512, 512)
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = 256, 256

    # Draw soft outer circular card
    draw.ellipse([26, 26, 486, 486], fill=(240, 253, 250, 255), outline=(0, 131, 143, 80), width=4)

    # Outer eye outline using polygon curve
    pts_top = []
    pts_bot = []
    steps = 100
    for i in range(steps + 1):
        t = i / steps
        # X: 70 -> 442
        x = 70 + t * 372
        # Y top arc (parabola peaking at y=110)
        y_top = 256 - 135 * 4 * t * (1 - t)
        # Y bot arc (parabola dipping at y=402)
        y_bot = 256 + 135 * 4 * t * (1 - t)
        pts_top.append((x, y_top))
        pts_bot.append((x, y_bot))

    eye_pts = pts_top + pts_bot[::-1]
    draw.polygon(eye_pts, fill=(255, 255, 255, 220), outline=(0, 131, 143, 255), width=16)

    # Retinal scan ring
    draw.ellipse([cx - 105, cy - 105, cx + 105, cy + 105], outline=(0, 172, 193, 160), width=6)

    # Iris
    draw.ellipse([cx - 76, cy - 76, cx + 76, cy + 76], fill=(2, 132, 199, 255), outline=(0, 131, 143, 255), width=4)

    # Pupil
    draw.ellipse([cx - 38, cy - 38, cx + 38, cy + 38], fill=(15, 23, 42, 255))

    # Optic highlights
    draw.ellipse([cx - 20, cy - 20, cx - 4, cy - 4], fill=(255, 255, 255, 230))
    draw.ellipse([cx + 12, cy + 12, cx + 22, cy + 22], fill=(255, 255, 255, 180))

    # Gold corner brackets
    gold = (245, 158, 11, 255)
    w = 5
    # Top-Left
    draw.line([(cx - 45, cy - 65), (cx - 60, cy - 65)], fill=gold, width=w)
    draw.line([(cx - 60, cy - 65), (cx - 60, cy - 50)], fill=gold, width=w)
    # Top-Right
    draw.line([(cx + 45, cy - 65), (cx + 60, cy - 65)], fill=gold, width=w)
    draw.line([(cx + 60, cy - 65), (cx + 60, cy - 50)], fill=gold, width=w)
    # Bottom-Left
    draw.line([(cx - 60, cy + 50), (cx - 60, cy + 65)], fill=gold, width=w)
    draw.line([(cx - 60, cy + 65), (cx - 45, cy + 65)], fill=gold, width=w)
    # Bottom-Right
    draw.line([(cx + 60, cy + 50), (cx + 60, cy + 65)], fill=gold, width=w)
    draw.line([(cx + 60, cy + 65), (cx + 45, cy + 65)], fill=gold, width=w)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    print(f"PNG logo saved to {output_path}")

if __name__ == "__main__":
    generate_svg_logo("assets/logo.svg")
    generate_png_logo("assets/logo.png")

