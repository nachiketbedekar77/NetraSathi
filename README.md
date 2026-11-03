# NetraSathi (नेत्रसाथी)
> **Explainable AI for Diabetic Retinopathy Screening in Rural India**  
> *A smartphone-friendly screening-support prototype for rural community health workers.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch CPU](https://img.shields.io/badge/PyTorch-CPU%20EfficientNet--B0-EE4C2C?logo=pytorch&logoColor=white)
![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM-00838F)

---

## 👁️ Overview

**NetraSathi** (*Netra* = Eye, *Sathi* = Companion) is an explainable Artificial Intelligence (XAI) screening tool engineered to empower frontline healthcare workers (ASHAs and ANMs) in remote Indian villages to screen diabetic patients for **Diabetic Retinopathy (DR)**.

In rural India, the ophthalmologist-to-patient ratio drops below **1 : 100,000**, leaving millions of diabetic individuals without access to regular retinal exams. NetraSathi bridges this gap through:
1. **Automated Image Quality Assessment:** Pre-inference blur and illumination checking using the Variance of Laplacian to advise retakes.
2. **Ben Graham Preprocessing:** Circular retinal masking, glare suppression, local Gaussian blur subtraction, and Green-channel CLAHE to enhance microvascular lesions.
3. **5-Level Severity Grading:** Classification into Grade 0 (No DR), Grade 1 (Mild), Grade 2 (Moderate), Grade 3 (Severe), and Grade 4 (Proliferative).
4. **Visual Explainability (Grad-CAM):** Gradient-weighted Class Activation Mapping computed from the convolutional feature maps, visually pinpointing microaneurysms, hemorrhages, and exudates.
5. **Instant Clinical PDF Reports:** Official downloadable reports containing patient demographics, side-by-side fundus and heatmap photos, severity ratings, and referral urgency triage.
6. **Multi-Language Accessibility:** Instant language toggle for **English**, **Hindi (हिंदी)**, and **Marathi (मराठी)**.

---

## 📸 Screenshots

| 1. Home & Key Rural Health Metrics | 2. Patient Registration |
|:---:|:---:|
| ![Home](assets/logo.png) | ![Registration](assets/logo.png) |

| 3. Retinal Screening & Grad-CAM Heatmap | 4. Patient History & Trajectory |
|:---:|:---:|
| Grade 0 to 4 Severity + Alpha Slider | Longitudinal DR Progression Charts |

---

## 🔬 Clinical Severity Scale & Referral Logic

| Grade | Clinical Condition | Referral & Care Recommendation |
|:---:|:---|:---|
| **0** | No Diabetic Retinopathy | Routine Care: Rescreen retinal fundus in 12 months. |
| **1** | Mild Non-Proliferative DR | Routine Care: Rescreen retinal fundus in 12 months. |
| **2** | Moderate Non-Proliferative DR | Semi-Urgent: Refer to eye specialist within 3 months. |
| **3** | Severe Non-Proliferative DR | **URGENT:** Immediate referral to vitreoretinal specialist. |
| **4** | Proliferative Diabetic Retinopathy | **URGENT:** Immediate referral to vitreoretinal specialist. |

---

## ⚖️ Honesty Statement & Demo Mode

- **Demo Weights Notice:** In this prototype demonstration, NetraSathi runs in **DEMO MODE** using an ImageNet-pretrained EfficientNet-B0 backbone. Predictions in demo mode are illustrative for college presentation and do not constitute a certified medical diagnosis.
- **Drop-in Real Weights:** A clinically trained model state dictionary can be placed at `weights/dr_model.pt`. When present, NetraSathi automatically switches from DEMO MODE to clinical inference.
- **Adding Real Fundus Images:** You can upload any standard retinal fundus photograph (JPEG/PNG) from open clinical datasets such as **Kaggle EyePACS**, **Messidor**, or **APTOS 2019 Blindness Detection**.

---

## 🚀 Local Installation & Run

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/<your-username>/netrasathi.git
cd netrasathi
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Streamlit Community Cloud Deployment

Follow these steps to deploy NetraSathi online:

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: NetraSathi prototype"
   git branch -M main
   git remote add origin https://github.com/<your-username>/netrasathi.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Sign in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
   - Click **"New app"**.
   - Select your repository: `<your-username>/netrasathi`.
   - Set **Branch:** `main`.
   - Set **Main file path:** `app.py`.
   - (Optional) In **Advanced settings**, set your custom URL subdomain (e.g. `netrasathi.streamlit.app`).
   - Click **"Deploy!"**.

> **Note on Storage:** Streamlit Community Cloud uses ephemeral disk storage. Registered patients and screening scans saved to the local SQLite database (`data/netrasathi.db`) persist during your active session.

---

## 👥 Project Team & Academic Credit

**College Demonstration Prototype**  
- **Institute:** Pune Vidyarthi Griha's College of Engineering and Technology and G. K. Pate (Wani) Institute of Management (**PVG COETM**)
- **Department:** Department of Information Technology
- **Class:** Second Year B.Tech (SY BTech IT)

### Student Engineers
1. **Sanskar Dingre**
2. **Avdhoot Jagtap**
3. **Atharva Gore**
4. **Nachiket Bedekar**

---

## 📄 License
Academic and educational prototype developed for collegiate demonstration under the Department of Information Technology, PVG COETM.
