"""
Browser Verification Script for NetraSathi.
Automates clicking through every screen and saving screenshots to the artifact directory.
"""
import os
import sys
import time
from playwright.sync_api import sync_playwright

ARTIFACTS_DIR = r"C:\Users\Nachiket\.gemini\antigravity\brain\1e7aecd8-2ad8-415c-b147-2a5e04f0150f"

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("Launching Chrome via Playwright...", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--window-size=1366,1200"]
        )
        context = browser.new_context(viewport={"width": 1366, "height": 1200})
        page = context.new_page()

        # ---------------------------------------------
        # 1. HOME SCREEN
        # ---------------------------------------------
        print("\n[1/5] Verifying Home Screen...", flush=True)
        page.goto("http://localhost:8501/?page=Home", timeout=30000)
        page.wait_for_selector(".ns-header", timeout=20000)
        time.sleep(2)

        shot1 = os.path.join(ARTIFACTS_DIR, "screen_1_home.png")
        page.screenshot(path=shot1, animations="disabled")
        print(f"Captured: {shot1}", flush=True)

        header_text = page.locator(".ns-header").inner_text()
        assert "NetraSathi" in header_text, "NetraSathi not in header!"
        print("Home page branding verified: NetraSathi present.", flush=True)

        # ---------------------------------------------
        # 2. PATIENT REGISTRATION SCREEN
        # ---------------------------------------------
        print("\n[2/5] Verifying Patient Registration...", flush=True)
        page.goto("http://localhost:8501/?page=Register", timeout=30000)
        page.wait_for_selector("input[aria-label='Full Name']", timeout=20000)
        time.sleep(1)

        # Fill in new patient
        page.locator("input[aria-label='Full Name']").fill("Shantabai Gaikwad")
        page.locator("input[aria-label='Phone Number']").fill("9822198765")
        page.locator("input[aria-label='Village / Tehsil']").fill("Jejuri, Taluka Purandar")
        page.locator("button:has-text('Register Patient')").click()
        time.sleep(2)

        shot2 = os.path.join(ARTIFACTS_DIR, "screen_2_register.png")
        page.screenshot(path=shot2, animations="disabled")
        print(f"Captured: {shot2}", flush=True)

        page_content = page.locator("body").inner_text()
        assert "Shantabai Gaikwad" in page_content or "NS-PAT-" in page_content
        print("Patient registration verified.", flush=True)

        # ---------------------------------------------
        # 3. SCREEN & EXPLAIN SCREEN
        # ---------------------------------------------
        print("\n[3/5] Verifying Screening & Grad-CAM Explainability...", flush=True)
        page.goto("http://localhost:8501/?page=Screen", timeout=30000)
        page.wait_for_selector(".result-container", timeout=30000)
        time.sleep(2)

        shot3 = os.path.join(ARTIFACTS_DIR, "screen_3_screening_gradcam.png")
        page.screenshot(path=shot3, animations="disabled")
        print(f"Captured: {shot3}", flush=True)

        res_text = page.locator(".result-container").inner_text()
        print(f"Severity Result Card: {res_text.strip().replace(chr(10), ' | ')}", flush=True)
        assert "Grade" in res_text

        # Scroll to PDF download button
        pdf_btn = page.locator("button:has-text('Download Official PDF')")
        pdf_btn.scroll_into_view_if_needed()
        time.sleep(1)
        assert pdf_btn.is_visible()
        print("Screening and Grad-CAM explainability verified. PDF button active.", flush=True)

        # ---------------------------------------------
        # 4. PATIENT HISTORY SCREEN
        # ---------------------------------------------
        print("\n[4/5] Verifying Patient History...", flush=True)
        page.goto("http://localhost:8501/?page=History", timeout=30000)
        time.sleep(3)

        shot4 = os.path.join(ARTIFACTS_DIR, "screen_4_history.png")
        page.screenshot(path=shot4, animations="disabled")
        print(f"Captured: {shot4}", flush=True)

        history_text = page.locator("body").inner_text()
        assert "SCN-" in history_text or "Grade" in history_text or "records" in history_text.lower()
        print("Patient history page verified.", flush=True)

        # ---------------------------------------------
        # 5. ABOUT PROJECT SCREEN
        # ---------------------------------------------
        print("\n[5/5] Verifying About Project Screen...", flush=True)
        page.goto("http://localhost:8501/?page=About", timeout=30000)
        page.wait_for_selector("text=About NetraSathi", timeout=20000)
        time.sleep(2)

        shot5 = os.path.join(ARTIFACTS_DIR, "screen_5_about.png")
        page.screenshot(path=shot5, animations="disabled")
        print(f"Captured: {shot5}", flush=True)

        about_text = page.locator("body").inner_text()
        for member in ["Sanskar Dingre", "Avdhoot Jagtap", "Atharva Gore", "Nachiket Bedekar"]:
            assert member in about_text, f"Missing team member: {member}"
        assert "Target Sensitivity" in about_text
        assert "PVG COETM" in about_text
        print("About page, target metrics, and team credits verified.", flush=True)

        browser.close()
        print("\n>>> ALL 5 SCREENS VERIFIED SUCCESSFULLY! <<<", flush=True)

if __name__ == "__main__":
    main()
