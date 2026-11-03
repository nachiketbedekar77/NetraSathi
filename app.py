"""
NetraSathi: Explainable AI for Diabetic Retinopathy Screening in Rural India
College Demonstration Prototype
PVG COETM, SY BTech IT, Department of Information Technology
Team: Sanskar Dingre, Avdhoot Jagtap, Atharva Gore, Nachiket Bedekar
"""
import os
import time
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

# Core modules
from core.db import (
    init_db, add_patient, get_patient, get_all_patients, search_patients,
    add_scan, get_patient_scans, DB_DIR, SCANS_DIR
)
from core.i18n import LANGUAGES, TRANSLATIONS, get_text
from core.preprocess import preprocess_retinal_image
from core.quality import assess_image_quality
from core.model import predict, GRADE_NAMES, load_model
from core.gradcam import generate_gradcam_overlay
from core.report import generate_pdf_report, get_referral_advice

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NetraSathi",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical medical theme (teal #00838F, slate, clean cards)
CUSTOM_CSS = """
<style>
    /* Global styling */
    .main {
        background-color: #F8FAFC;
    }
    
    /* Header card */
    .ns-header {
        background: linear-gradient(135deg, #00838F 0%, #00695C 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .ns-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .ns-header p {
        color: #E0F2FE !important;
        font-size: 1.05rem;
        margin: 0.2rem 0 0 0;
    }

    /* Demo Mode Banner */
    .ns-demo-banner {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 0.75rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        color: #991B1B;
        font-weight: 600;
        font-size: 0.92rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Stat Cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .stat-num {
        font-size: 2rem;
        font-weight: 800;
        color: #00838F;
        margin-bottom: 0.2rem;
    }
    .stat-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.3;
    }

    /* Grade Result Badges */
    .grade-badge-0 { background-color: #DCFCE7; color: #166534; border: 2px solid #22C55E; }
    .grade-badge-1 { background-color: #FEF9C3; color: #854D0E; border: 2px solid #EAB308; }
    .grade-badge-2 { background-color: #FFEDD5; color: #9A3412; border: 2px solid #F97316; }
    .grade-badge-3 { background-color: #FEE2E2; color: #991B1B; border: 2px solid #EF4444; }
    .grade-badge-4 { background-color: #7F1D1D; color: #FEF2F2; border: 2px solid #991B1B; }

    .result-container {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .result-subtitle {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    /* Active patient pill */
    .active-patient-pill {
        display: inline-block;
        background: #E0F2FE;
        color: #0369A1;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 1rem;
        border: 1px solid #BAE6FD;
    }

    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        border-color: #00838F;
        color: #00838F;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "active_patient_id" not in st.session_state:
    st.session_state["active_patient_id"] = None
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Home"

# Query parameter routing support
try:
    if "page" in st.query_params:
        p_param = st.query_params["page"]
        if p_param in ["Home", "Register", "Screen", "History", "About"]:
            st.session_state["nav_page"] = p_param
except Exception:
    pass

# Pre-populate sample patient if empty
all_pts = get_all_patients()
if len(all_pts) == 0:
    p1 = add_patient(
        name="Sunita Deshmukh",
        age=54,
        sex="Female",
        phone="9823456789",
        village="Saswad, Pune",
        years_diabetes=6.5
    )
    p2 = add_patient(
        name="Rameshwar Patil",
        age=61,
        sex="Male",
        phone="9876543210",
        village="Shirwal, Satara",
        years_diabetes=12.0
    )
    st.session_state["active_patient_id"] = p1
elif st.session_state["active_patient_id"] is None:
    st.session_state["active_patient_id"] = all_pts[0]["id"]

lang = st.session_state["lang"]

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    # Logo & Title
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    
    st.markdown(f"### **{get_text('app_name', lang)}**")
    st.caption(get_text("tagline", lang))
    st.markdown("---")

    # Language Toggle
    st.markdown("🌐 **Language / भाषा**")
    selected_lang_name = st.selectbox(
        "Select Language",
        options=list(LANGUAGES.values()),
        index=list(LANGUAGES.keys()).index(st.session_state["lang"]),
        label_visibility="collapsed"
    )
    # Reverse lookup key
    for k, v in LANGUAGES.items():
        if v == selected_lang_name:
            if st.session_state["lang"] != k:
                st.session_state["lang"] = k
                st.rerun()

    st.markdown("---")

    # Navigation Menu
    nav_options = [
        get_text("nav_home", lang),
        get_text("nav_register", lang),
        get_text("nav_screen", lang),
        get_text("nav_history", lang),
        get_text("nav_about", lang)
    ]
    
    # Internal page mapping
    internal_pages = ["Home", "Register", "Screen", "History", "About"]
    current_idx = 0
    if st.session_state["nav_page"] in internal_pages:
        current_idx = internal_pages.index(st.session_state["nav_page"])

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=current_idx,
        label_visibility="collapsed"
    )
    st.session_state["nav_page"] = internal_pages[nav_options.index(selected_nav)]

    st.markdown("---")
    
    # Active Patient Sidebar Indicator
    if st.session_state["active_patient_id"]:
        pt = get_patient(st.session_state["active_patient_id"])
        if pt:
            st.markdown("👤 **Active Patient**")
            st.write(f"**{pt['name']}** ({pt['age']}y, {pt['sex']})")
            st.caption(f"📍 {pt['village']} | ⏱️ {pt['years_diabetes']} yrs diabetic")
            if st.button("Change Patient", use_container_width=True):
                st.session_state["nav_page"] = "Register"
                st.rerun()
    else:
        st.info("No active patient selected.")

    st.markdown("---")
    st.caption("🏛️ **PVG COETM** | SY BTech IT\n\nDepartment of Information Technology")


# ---------------------------------------------------------
# Top Header on Every Page
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="ns-header">
        <h1>👁️ {get_text('app_name', lang)}</h1>
        <p>{get_text('tagline', lang)}</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------
if st.session_state["nav_page"] == "Home":
    st.markdown(
        f"""
        <div class="ns-demo-banner">
            ⚠️ {get_text('demo_mode_banner', lang)}
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.8, 1.2])

    with col1:
        st.markdown(
            """
            ### Empowering Rural Healthcare Workers with Explainable AI
            **NetraSathi** is a smartphone-compatible screening-support platform engineered to detect 
            **Diabetic Retinopathy (DR)** early in remote and underserved villages across India. 

            By pairing deep convolutional neural feature extraction (**EfficientNet-B0**) with 
            gradient-weighted class activation mapping (**Grad-CAM**), NetraSathi bridges the trust gap 
            between AI predictions and primary healthcare providers (ASHA and ANM workers).
            """
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        # Big Start Screening CTA
        if st.button(get_text("start_screening_btn", lang), type="primary", use_container_width=True):
            st.session_state["nav_page"] = "Screen"
            st.rerun()

    with col2:
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, caption="NetraSathi: Eye Screening Companion", use_container_width=True)

    st.markdown("---")

    # Key Statistics Grid
    st.markdown("### The Rural Indian Eye Care Crisis")
    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-num">77M+</div>
                <div class="stat-desc">Diabetic individuals in India requiring annual dilated retinal fundus evaluation.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s_col2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-num">1 : 100K</div>
                <div class="stat-desc">Ophthalmologist-to-patient ratio in rural India, creating an acute screening barrier.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s_col3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-num">80%+</div>
                <div class="stat-desc">Of vision impairment from diabetic retinopathy is preventable if triaged in early stages.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # 4-Step Clinical Workflow
    st.markdown("### How NetraSathi Works in Primary Health Centres")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.info("#### 1. Registration\nASHA worker logs patient identity, village, and diabetes duration.")
    with w2:
        st.info("#### 2. Photo & Triage\nUpload or capture retinal photo with automated blur/quality checks.")
    with w3:
        st.info("#### 3. AI Grading\nBen Graham cleanup + 5-grade severity rating with confidence metric.")
    with w4:
        st.info("#### 4. Grad-CAM & PDF\nVisual explainability heatmap & printable clinical referral PDF.")


# ---------------------------------------------------------
# PAGE 2: PATIENT REGISTRATION
# ---------------------------------------------------------
elif st.session_state["nav_page"] == "Register":
    st.subheader(f"📋 {get_text('patient_reg_title', lang)}")

    tab1, tab2 = st.tabs(["➕ Register New Patient", "🔍 Search & Select Existing Patient"])

    with tab1:
        with st.form("patient_registration_form", clear_on_submit=True):
            r1, r2 = st.columns(2)
            with r1:
                name = st.text_input(get_text("patient_name", lang), placeholder="e.g. Anand Kulkarni")
                age = st.number_input(get_text("patient_age", lang), min_value=1, max_value=110, value=52)
                sex = st.selectbox(
                    get_text("patient_sex", lang),
                    [get_text("sex_male", lang), get_text("sex_female", lang), get_text("sex_other", lang)]
                )
            with r2:
                phone = st.text_input(get_text("patient_phone", lang), placeholder="e.g. 9822012345")
                village = st.text_input(get_text("patient_village", lang), placeholder="e.g. Saswad, Taluka Purandar")
                years_diabetes = st.number_input(
                    get_text("patient_diabetes_years", lang),
                    min_value=0.0, max_value=50.0, value=5.0, step=0.5
                )

            submitted = st.form_submit_button(get_text("btn_save_patient", lang), type="primary")
            if submitted:
                if not name.strip():
                    st.error("Please enter a valid patient name.")
                else:
                    new_id = add_patient(name, age, sex, phone, village, years_diabetes)
                    st.session_state["active_patient_id"] = new_id
                    st.success(f"{get_text('patient_saved_msg', lang)} NS-PAT-{new_id:04d}")
                    st.info("Patient set as active! You can now proceed directly to Screening.")

    with tab2:
        search_query = st.text_input(get_text("search_patient", lang), placeholder="Search by name, village, or phone...")
        matched_patients = search_patients(search_query)

        if matched_patients:
            for p in matched_patients:
                with st.container():
                    c_info, c_btn = st.columns([4, 1.2])
                    with c_info:
                        is_active = (p["id"] == st.session_state["active_patient_id"])
                        active_mark = " ⭐ **[ACTIVE]**" if is_active else ""
                        st.markdown(
                            f"**NS-PAT-{p['id']:04d}: {p['name']}**{active_mark}  \n"
                            f"Age: {p['age']} | Sex: {p['sex']} | Village: {p['village']} | Phone: {p['phone']} | "
                            f"Diabetes: {p['years_diabetes']} yrs"
                        )
                    with c_btn:
                        if st.button("Select Patient", key=f"sel_p_{p['id']}", use_container_width=True):
                            st.session_state["active_patient_id"] = p["id"]
                            st.session_state["nav_page"] = "Screen"
                            st.rerun()
                    st.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)
        else:
            st.warning(get_text("no_patient_found", lang))


# ---------------------------------------------------------
# PAGE 3: SCREEN & EXPLAIN
# ---------------------------------------------------------
elif st.session_state["nav_page"] == "Screen":
    st.subheader(f"🔬 {get_text('screening_title', lang)}")

    # Check active patient
    patient = None
    if st.session_state["active_patient_id"]:
        patient = get_patient(st.session_state["active_patient_id"])

    if not patient:
        st.warning("⚠️ No active patient selected. Please select or register a patient first.")
        if st.button("Go to Patient Registration", type="primary"):
            st.session_state["nav_page"] = "Register"
            st.rerun()
        st.stop()

    # Active Patient Badge & Switcher
    col_pt1, col_pt2 = st.columns([2.5, 1.5])
    with col_pt1:
        st.markdown(
            f"""
            <div class="active-patient-pill">
                👤 Active Patient: <b>{patient['name']}</b> (NS-PAT-{patient['id']:04d}) | 
                {patient['age']} yrs, {patient['sex']} | 📍 {patient['village']}
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_pt2:
        all_p = get_all_patients()
        p_labels = [f"NS-PAT-{p['id']:04d}: {p['name']} ({p['village']})" for p in all_p]
        curr_idx = 0
        for i, p in enumerate(all_p):
            if p["id"] == patient["id"]:
                curr_idx = i
                break
        sel_label = st.selectbox(
            "Switch Active Patient",
            p_labels,
            index=curr_idx,
            label_visibility="collapsed"
        )
        sel_p_id = all_p[p_labels.index(sel_label)]["id"]
        if sel_p_id != st.session_state["active_patient_id"]:
            st.session_state["active_patient_id"] = sel_p_id
            st.rerun()

    # Screening Settings Column
    cfg_col1, cfg_col2 = st.columns([1, 1.5])
    with cfg_col1:
        eye_choice = st.radio(
            get_text("select_eye", lang),
            [get_text("eye_right", lang), get_text("eye_left", lang)],
            horizontal=True
        )

    with cfg_col2:
        source_mode = st.radio(
            get_text("input_method", lang),
            [get_text("input_sample", lang), get_text("input_upload", lang), get_text("input_camera", lang)],
            horizontal=True
        )

    image_to_process = None

    # Source 1: Verified Sample Images
    if source_mode == get_text("input_sample", lang):
        sample_choice = st.selectbox(
            "Select Benchmark Sample Retinal Image",
            [
                get_text("sample_normal", lang),
                get_text("sample_moderate", lang),
                get_text("sample_severe", lang)
            ]
        )
        if sample_choice == get_text("sample_normal", lang):
            s_file = os.path.join(os.path.dirname(__file__), "samples", "sample_normal.jpg")
        elif sample_choice == get_text("sample_moderate", lang):
            s_file = os.path.join(os.path.dirname(__file__), "samples", "sample_moderate.jpg")
        else:
            s_file = os.path.join(os.path.dirname(__file__), "samples", "sample_severe.jpg")

        if os.path.exists(s_file):
            image_to_process = Image.open(s_file).convert("RGB")
        else:
            st.error(f"Sample image not found at {s_file}")

    # Source 2: File Upload
    elif source_mode == get_text("input_upload", lang):
        uploaded_file = st.file_uploader(
            "Upload fundus photo (JPEG/PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload retinal fundus photograph captured from clinical camera or smartphone ophthalmoscope attachment."
        )
        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error("Error reading uploaded image file.")

    # Source 3: Camera Input
    else:
        cam_pic = st.camera_input("Capture retinal fundus via smartphone lens attachment")
        if cam_pic is not None:
            try:
                image_to_process = Image.open(cam_pic).convert("RGB")
            except Exception as e:
                st.error("Error capturing image from camera.")

    # If an image is available, run the pipeline
    if image_to_process is not None:
        st.markdown("---")

        # Step 1: Preprocessing & Cleanup
        with st.spinner("Processing retinal optics (Ben Graham Illumination & CLAHE)..."):
            raw_bgr, cleaned_bgr, mask, circle_info = preprocess_retinal_image(image_to_process)
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

        # Step 2: Quality Assessment
        quality = assess_image_quality(raw_bgr, mask, circle_info)

        # Image Quality Banner
        st.markdown(f"#### 🔎 {get_text('quality_title', lang)}")
        q_cols = st.columns([1, 1, 1, 2])
        q_cols[0].metric(get_text("quality_score_label", lang), f"{quality['quality_score']:.2f} / 1.00")
        q_cols[1].metric("Blur Variance", f"{quality['blur_score']}")
        q_cols[2].metric("Luminance", f"{quality['brightness_score']} / 255")

        if quality["is_acceptable"]:
            q_cols[3].success(f"✅ {get_text('quality_good', lang)}")
        else:
            for w in quality["warnings"]:
                q_cols[3].warning(f"⚠️ {w}")

        # Show Before / After Side-by-Side
        st.markdown(f"#### 🧼 {get_text('clean_title', lang)}")
        clean_col1, clean_col2 = st.columns(2)
        with clean_col1:
            st.image(raw_rgb, caption=get_text("clean_before", lang), use_container_width=True)
        with clean_col2:
            st.image(cleaned_rgb, caption=get_text("clean_after", lang), use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Screening Execution Button
        run_analysis = True
        if not quality["is_acceptable"]:
            st.warning("Image quality is below ideal threshold. You can choose to proceed anyway.")
            run_analysis = st.checkbox("Analyse anyway despite quality warning", value=True)

        if run_analysis:
            # Model Inference
            with st.spinner("Running EfficientNet-B0 inference & Grad-CAM backward pass..."):
                grade, probs, latency, is_demo = predict(raw_rgb)
                grade_name = GRADE_NAMES[grade]
                confidence = probs[grade]

            # DEMO MODE Persistent Banner
            if is_demo:
                st.markdown(
                    f"""
                    <div class="ns-demo-banner">
                        ⚠️ {get_text('demo_mode_banner', lang)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Result Card
            ref_title, ref_desc, badge_color = get_referral_advice(grade)
            st.markdown(f"#### 🏆 {get_text('result_card_title', lang)}")

            badge_class = f"grade-badge-{grade}"
            st.markdown(
                f"""
                <div class="result-container {badge_class}">
                    <div class="result-title">{grade_name}</div>
                    <div class="result-subtitle">Model Confidence: {confidence:.1%} | Latency: {latency:.3f}s (CPU)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; margin-top: 0.6rem;">{ref_title}</div>
                    <div style="font-size: 0.95rem; margin-top: 0.3rem;">{ref_desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Low Confidence Warning
            if confidence < 0.60:
                st.warning(f"⚠️ {get_text('low_confidence_warning', lang)}")

            # Probabilities Bar Chart
            st.markdown(f"#### 📊 {get_text('inference_title', lang)}")
            chart_col, cam_col = st.columns([1, 1])

            with chart_col:
                # Custom horizontal bar plot using matplotlib
                fig, ax = plt.subplots(figsize=(6, 3.8))
                y_positions = np.arange(5)
                short_grades = ["0: No DR", "1: Mild", "2: Moderate", "3: Severe", "4: Proliferative"]
                bar_colors = ["#22C55E", "#EAB308", "#F97316", "#EF4444", "#7F1D1D"]

                bars = ax.barh(y_positions, probs, color=bar_colors, height=0.6, edgecolor="#334155")
                ax.set_yticks(y_positions)
                ax.set_yticklabels(short_grades, fontsize=10, fontweight="bold")
                ax.set_xlim(0, 1.0)
                ax.set_xlabel("Probability", fontsize=10)
                ax.invert_yaxis()  # Grade 0 on top

                for bar, p in zip(bars, probs):
                    ax.text(
                        bar.get_width() + 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{p:.1%}",
                        va="center",
                        fontsize=9,
                        fontweight="bold"
                    )

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

            with cam_col:
                st.markdown(f"**{get_text('heatmap_title', lang)}**")
                st.caption(get_text("heatmap_desc", lang))

                alpha_slider = st.slider(
                    get_text("heatmap_slider", lang),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.45,
                    step=0.05
                )

                overlay_rgb, heatmap = generate_gradcam_overlay(
                    raw_rgb, mask, target_class=grade, alpha=alpha_slider
                )
                st.image(overlay_rgb, caption=f"Grad-CAM Heatmap (Alpha={alpha_slider:.2f})", use_container_width=True)

            st.markdown("---")

            # PDF Report Generation & Auto-Save
            st.markdown("#### 📄 Clinical Report & Export")
            scan_meta = {
                "eye": eye_choice,
                "grade": grade,
                "grade_name": grade_name,
                "confidence": confidence,
                "quality_score": quality["quality_score"],
                "blur_score": quality["blur_score"],
                "referral_advice": f"{ref_title}: {ref_desc}"
            }

            orig_pil = Image.fromarray(raw_rgb)
            overlay_pil = Image.fromarray(overlay_rgb)

            # Generate PDF bytes
            pdf_bytes = generate_pdf_report(
                patient_data=patient,
                scan_data=scan_meta,
                orig_img_pil=orig_pil,
                heatmap_img_pil=overlay_pil,
                is_demo=is_demo
            )

            # Save scan record to SQLite automatically
            # Check if this scan was already saved in this session
            scan_saved_key = f"saved_scan_{patient['id']}_{grade}_{time.time() // 60}"
            if scan_saved_key not in st.session_state:
                # Save scan files locally
                scan_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_save_path = os.path.join(SCANS_DIR, f"p{patient['id']}_{scan_ts}_orig.jpg")
                cam_save_path = os.path.join(SCANS_DIR, f"p{patient['id']}_{scan_ts}_cam.jpg")
                orig_pil.save(img_save_path, quality=90)
                overlay_pil.save(cam_save_path, quality=90)

                add_scan(
                    patient_id=patient["id"],
                    eye=eye_choice,
                    grade=grade,
                    grade_name=grade_name,
                    confidence=confidence,
                    quality_score=quality["quality_score"],
                    blur_score=quality["blur_score"],
                    brightness_score=quality["brightness_score"],
                    image_path=img_save_path,
                    heatmap_path=cam_save_path,
                    referral_advice=ref_title
                )
                st.session_state[scan_saved_key] = True

            clean_pname = "".join(c for c in patient["name"] if c.isalnum() or c == "_")
            report_filename = f"NetraSathi_Report_{clean_pname}_{datetime.now().strftime('%Y%m%d')}.pdf"

            st.download_button(
                label=get_text("btn_download_pdf", lang),
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )


# ---------------------------------------------------------
# PAGE 4: PATIENT HISTORY
# ---------------------------------------------------------
elif st.session_state["nav_page"] == "History":
    st.subheader(f"📈 {get_text('history_title', lang)}")

    all_patients = get_all_patients()
    if not all_patients:
        st.info("No patients registered yet.")
        st.stop()

    # Patient Selector for History
    pt_options = {f"NS-PAT-{p['id']:04d}: {p['name']} ({p['village']})": p["id"] for p in all_patients}
    
    # Default to current active patient
    current_key_idx = 0
    if st.session_state["active_patient_id"]:
        for idx, (label, pid) in enumerate(pt_options.items()):
            if pid == st.session_state["active_patient_id"]:
                current_key_idx = idx
                break

    selected_pt_label = st.selectbox(
        "Select Patient to View Longitudinal History",
        options=list(pt_options.keys()),
        index=current_key_idx
    )
    selected_pid = pt_options[selected_pt_label]
    patient = get_patient(selected_pid)

    scans = get_patient_scans(selected_pid)

    if not scans:
        st.info(get_text("history_no_scans", lang))
    else:
        st.write(f"Found **{len(scans)}** recorded screening scans for **{patient['name']}**.")

        # Longitudinal Progression Chart
        if len(scans) >= 2:
            st.markdown("#### Longitudinal DR Severity Progression")
            # Parse dates and grades (oldest first for trend)
            scans_chronological = sorted(scans, key=lambda s: s["created_at"])
            dates = [s["created_at"][:10] for s in scans_chronological]
            grades = [s["grade"] for s in scans_chronological]

            fig_hist, ax_hist = plt.subplots(figsize=(8, 2.8))
            ax_hist.plot(dates, grades, marker="o", color="#00838F", linewidth=2.5, markersize=8)
            ax_hist.set_yticks([0, 1, 2, 3, 4])
            ax_hist.set_yticklabels(["0: None", "1: Mild", "2: Mod", "3: Sev", "4: Prolif"])
            ax_hist.set_ylim(-0.3, 4.3)
            ax_hist.grid(axis="y", linestyle="--", alpha=0.5)
            ax_hist.set_title(f"Severity Trajectory: {patient['name']}", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_hist)

        # Scans Table
        st.markdown("#### Screening Records Table")
        table_rows = []
        for s in scans:
            table_rows.append({
                "Scan ID": f"SCN-{s['id']:04d}",
                "Date": s["created_at"][:16].replace("T", " "),
                "Eye": s["eye"],
                "Grade": s["grade_name"],
                "Confidence": f"{s['confidence']:.1%}",
                "Quality": f"{s['quality_score']:.2f}",
                "Referral Advice": s["referral_advice"]
            })
        st.table(table_rows)

        # Thumbnails Gallery
        st.markdown("#### Retinal Image & Heatmap Gallery")
        gallery_cols = st.columns(min(len(scans), 3))
        for idx, s in enumerate(scans[:3]):
            with gallery_cols[idx]:
                st.markdown(f"**{s['created_at'][:10]} | {s['eye']}**")
                if os.path.exists(s["heatmap_path"]):
                    st.image(s["heatmap_path"], caption=f"{s['grade_name']} ({s['confidence']:.0%})", use_container_width=True)
                elif os.path.exists(s["image_path"]):
                    st.image(s["image_path"], caption=f"{s['grade_name']} ({s['confidence']:.0%})", use_container_width=True)


# ---------------------------------------------------------
# PAGE 5: ABOUT
# ---------------------------------------------------------
elif st.session_state["nav_page"] == "About":
    st.subheader(f"ℹ️ {get_text('nav_about', lang)}")

    st.markdown(
        """
        ### About NetraSathi (नेत्रसाथी)
        **NetraSathi** is an explainable Artificial Intelligence (XAI) prototype designed specifically 
        to empower rural community health workers (ASHAs and ANMs) in India to conduct frontline screening 
        for **Diabetic Retinopathy (DR)**.
        """
    )

    st.markdown("---")

    # Architecture Overview
    st.markdown("### System Architecture")
    st.markdown(
        """
        ```
        +-------------------------+      +-------------------------------+      +-------------------------+
        |   Rural Health Worker   | ---> |     NetraSathi AI Engine      | ---> |   Primary Health Centre |
        | Smartphone Retinal Lens |      |   EfficientNet-B0 + Grad-CAM  |      |  Patient Database (SQL) |
        +-------------------------+      +-------------------------------+      +-------------------------+
                     |                                   |                                   |
                     v                                   v                                   v
             Camera / Upload                    Ben Graham Preprocessing                 PDF Report
             Quality Filtering                  5-Class DR Severity Grade              Referral Advice
        ```
        """
    )

    # 3 Research Gaps
    st.markdown("### Research Gaps Addressed")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.error(
            "#### Gap 1: Black-Box Skepticism\n"
            "Traditional deep learning diagnostic systems provide opaque classification scores without visual verification. "
            "NetraSathi integrates **Grad-CAM heatmaps** into 100% of screenings to pinpoint microaneurysms and exudates."
        )
    with g2:
        st.warning(
            "#### Gap 2: GPU Dependency\n"
            "Clinical DR models typically rely on heavy server GPUs inaccessible to rural primary health centres. "
            "NetraSathi uses an optimized **EfficientNet-B0** architecture delivering sub-second inference on standard CPUs."
        )
    with g3:
        st.info(
            "#### Gap 3: Specialist Scarcity\n"
            "With an eye doctor ratio dropping below 1:100,000 in rural India, patients miss early detection windows. "
            "NetraSathi provides automated, clear referral triage (12-month rescreen vs immediate hospital referral)."
        )

    st.markdown("---")

    # Target Metrics
    st.markdown("### Project Target Metrics")
    st.caption("Note: The following performance indicators are engineering design TARGETS for full clinical validation:")
    tm1, tm2, tm3 = st.columns(3)
    with tm1:
        st.metric("Target Sensitivity / Specificity", "> 92.0%", help="Target metric for clinical validation")
    with tm2:
        st.metric("Target CPU Inference Latency", "< 5.0 seconds", help="Target metric on commodity dual-core CPUs")
    with tm3:
        st.metric("Target Heatmap Generation", "100% of scans", help="Target metric for explainability compliance")

    st.markdown("---")

    # Team & Academic Credit
    st.markdown("### Academic Demonstration Team")
    st.markdown(
        """
        - **College:** Pune Vidyarthi Griha's College of Engineering and Technology and G. K. Pate (Wani) Institute of Management (**PVG COETM**)
        - **Department:** Department of Information Technology
        - **Class:** Second Year B.Tech (SY BTech IT)
        - **Student Engineers:**
          1. **Sanskar Dingre**
          2. **Avdhoot Jagtap**
          3. **Atharva Gore**
          4. **Nachiket Bedekar**
        """
    )

    st.markdown("---")

    # Limitations & Honesty Statement
    st.markdown("### Clinical Limitations & Honesty Statement")
    st.markdown(
        """
        - **Model Weights & Training:** In this prototype demonstration, NetraSathi runs in **DEMO MODE** using an ImageNet-pretrained backbone. Clinically trained weights can be dropped in seamlessly at `weights/dr_model.pt`.
        - **Imaging Conditions:** Standard deep learning DR models are trained on high-end mydriatic fundus cameras (such as Topcon or Zeiss). Smartphone lenses introduce optical reflections and varying fields-of-view that require rigorous prospective field calibration.
        - **Not a Primary Diagnosis:** NetraSathi is engineered strictly as a screening decision-support aid for frontline health workers, not as a replacement for comprehensive biomicroscopy by an ophthalmologist.
        """
    )

    st.markdown("---")

    # Future Scope
    st.markdown("### Future Roadmap")
    st.markdown(
        """
        - **ABHA Integration:** Direct linking with Ayushman Bharat Digital Health ID for longitudinal tracking across national healthcare records.
        - **Multi-Disease Expansion:** Extending convolutional heads for simultaneous detection of Glaucoma (cup-to-disc ratio) and Hypertensive Retinopathy.
        - **Edge Deployment:** Exporting the quantized model to ONNX Runtime and TFLite for completely offline smartphone operation without internet connectivity.
        """
    )


# ---------------------------------------------------------
# Persistent Global Footer
# ---------------------------------------------------------
st.markdown("<br/><br/>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align: center; color: #64748B; font-size: 0.82rem; border-top: 1px solid #E2E8F0; padding-top: 1rem;">
        {get_text('disclaimer_footer', lang)} | PVG COETM SY BTech IT
    </div>
    """,
    unsafe_allow_html=True
)
