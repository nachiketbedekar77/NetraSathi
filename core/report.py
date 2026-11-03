"""
Clinical Screening PDF Report Generator for NetraSathi.
Generates an official diagnostic screening report using ReportLab with:
- NetraSathi branding and logo
- Patient demographic metadata
- Side-by-side retinal photo & Grad-CAM explainability heatmap
- Severity grade, confidence, referral recommendations
- Persistent DEMO MODE notice and clinical disclaimer
"""
import os
import io
from datetime import datetime
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")

def get_referral_advice(grade: int) -> tuple[str, str, colors.Color]:
    """Returns (summary, details, badge_color)."""
    if grade in (0, 1):
        return (
            "Routine Care: Rescreen in 12 Months",
            "No urgent vision-threatening microvascular pathology detected. Recommend standard annual retinal examination and strict glycemic control.",
            colors.HexColor("#16A34A") # Green
        )
    elif grade == 2:
        return (
            "Semi-Urgent: Refer to Specialist within 3 Months",
            "Moderate microvascular damage with microaneurysms and exudates present. Specialist dilated examination needed to assess macular edema risk.",
            colors.HexColor("#D97706") # Amber/Orange
        )
    else:
        return (
            "URGENT: Immediate Ophthalmologist Referral",
            "Critical sight-threatening pathology (severe hemorrhages, cotton wool spots, or neovascularization). High imminent risk of vitreous hemorrhage or tractional retinal detachment.",
            colors.HexColor("#DC2626") # Red
        )

def generate_pdf_report(
    patient_data: dict,
    scan_data: dict,
    orig_img_pil: Image.Image,
    heatmap_img_pil: Image.Image,
    is_demo: bool = True
) -> bytes:
    """
    Generates a PDF bytes object for downloading.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#00838F')
    )
    tagline_style = ParagraphStyle(
        'Tagline',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569')
    )
    section_heading = ParagraphStyle(
        'SectionHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B')
    )
    banner_style = ParagraphStyle(
        'Banner',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#991B1B'),
        alignment=1 # Center
    )
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#64748B'),
        alignment=1 # Center
    )

    story = []

    # 1. Header with Logo & Title
    header_data = []
    logo_flowable = None
    if os.path.exists(LOGO_PATH):
        try:
            logo_flowable = RLImage(LOGO_PATH, width=56, height=56)
        except Exception:
            logo_flowable = Paragraph("<b>👁️ NetraSathi</b>", title_style)
    else:
        logo_flowable = Paragraph("<b>👁️ NetraSathi</b>", title_style)

    title_block = [
        Paragraph("<b>NetraSathi</b>", title_style),
        Paragraph("Explainable AI for Diabetic Retinopathy Screening in Rural India", tagline_style),
        Paragraph("College Demonstration Prototype | PVG COETM, SY BTech IT", tagline_style)
    ]
    header_table = Table([[logo_flowable, title_block]], colWidths=[65, 475])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # 2. Demo Mode Banner
    if is_demo:
        demo_box = Table(
            [[Paragraph("DEMO MODE: NetraSathi's model is not clinically trained. Grades are illustrative only. Not a medical diagnosis.", banner_style)]],
            colWidths=[540]
        )
        demo_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEE2E2')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#EF4444')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(demo_box)
        story.append(Spacer(1, 10))

    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#00838F'), spaceAfter=8))

    # 3. Patient Demographic Information Table
    story.append(Paragraph("PATIENT & EXAMINATION DETAILS", section_heading))

    scan_date = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    p_id = f"NS-PAT-{patient_data.get('id', 1):04d}"
    p_name = patient_data.get('name', 'Anonymous')
    p_age = f"{patient_data.get('age', 'N/A')} yrs"
    p_sex = patient_data.get('sex', 'N/A')
    p_village = patient_data.get('village', 'N/A')
    p_phone = patient_data.get('phone', 'N/A')
    p_diabetes = f"{patient_data.get('years_diabetes', 0)} years"
    p_eye = scan_data.get('eye', 'Right Eye (OD)')

    demographics_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {p_id}", body_style),
            Paragraph(f"<b>Examination Date:</b> {scan_date}", body_style)
        ],
        [
            Paragraph(f"<b>Full Name:</b> {p_name}", body_style),
            Paragraph(f"<b>Examined Eye:</b> <b>{p_eye}</b>", body_style)
        ],
        [
            Paragraph(f"<b>Age / Sex:</b> {p_age} / {p_sex}", body_style),
            Paragraph(f"<b>Village / Location:</b> {p_village}", body_style)
        ],
        [
            Paragraph(f"<b>Contact Phone:</b> {p_phone}", body_style),
            Paragraph(f"<b>Diabetes Duration:</b> {p_diabetes}", body_style)
        ]
    ]
    dem_table = Table(demographics_data, colWidths=[270, 270])
    dem_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(dem_table)
    story.append(Spacer(1, 10))

    # 4. Side-by-Side Images: Original and Grad-CAM Heatmap
    story.append(Paragraph("RETINAL IMAGING & EXPLAINABILITY EVIDENCE (GRAD-CAM)", section_heading))

    # Save temporary in-memory images for ReportLab
    orig_buf = io.BytesIO()
    orig_img_pil.save(orig_buf, format="JPEG", quality=90)
    orig_buf.seek(0)
    rl_orig = RLImage(orig_buf, width=190, height=190)

    cam_buf = io.BytesIO()
    heatmap_img_pil.save(cam_buf, format="JPEG", quality=90)
    cam_buf.seek(0)
    rl_cam = RLImage(cam_buf, width=190, height=190)

    img_table_data = [
        [rl_orig, rl_cam],
        [
            Paragraph("<b>Raw Retinal Fundus Photograph</b>", body_style),
            Paragraph("<b>Explainable Grad-CAM Heatmap Overlay</b>", body_style)
        ]
    ]
    img_table = Table(img_table_data, colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # 5. Diagnostic Severity Assessment & Recommendation Card
    story.append(Paragraph("AI SCREENING ASSESSMENT & REFERRAL GUIDANCE", section_heading))

    grade = scan_data.get('grade', 0)
    grade_name = scan_data.get('grade_name', 'Grade 0: No DR')
    confidence = scan_data.get('confidence', 0.0)
    quality_score = scan_data.get('quality_score', 1.0)
    ref_title, ref_desc, badge_color = get_referral_advice(grade)

    assessment_data = [
        [
            Paragraph("<b>Assigned Severity Grade:</b>", body_style),
            Paragraph(f"<b><font size=12 color='{badge_color.hexval()}'>{grade_name}</font></b>", body_style)
        ],
        [
            Paragraph("<b>Confidence / Quality:</b>", body_style),
            Paragraph(f"Model Confidence: <b>{confidence:.1%}</b> | Image Quality: <b>{quality_score:.2f}/1.00</b>", body_style)
        ],
        [
            Paragraph("<b>Action Recommendation:</b>", body_style),
            Paragraph(f"<b><font color='{badge_color.hexval()}'>{ref_title}</font></b><br/>{ref_desc}", body_style)
        ]
    ]
    ass_table = Table(assessment_data, colWidths=[160, 380])
    ass_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94A3B8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(ass_table)
    story.append(Spacer(1, 14))

    # 6. Disclaimer & Footer
    disclaimer_text = (
        "<b>CLINICAL DISCLAIMER:</b> NetraSathi is a mobile-oriented screening decision-support aid designed for "
        "rural healthcare workers and college research demonstration. It does not replace a comprehensive ophthalmic "
        "biomicroscopic examination. Always confirm all findings with a qualified vitreoretinal surgeon or ophthalmologist."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#64748B'))))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=6))
    story.append(Paragraph("Generated by NetraSathi - screening aid, not a diagnosis | PVG COETM SY BTech IT", footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

