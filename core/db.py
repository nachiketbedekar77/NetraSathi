"""
Database management module for NetraSathi using SQLite.
Stores patient registrations and screening records locally.
"""
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "netrasathi.db")
SCANS_DIR = os.path.join(DB_DIR, "scans")

def get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(SCANS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize SQLite database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Patients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            sex TEXT NOT NULL,
            phone TEXT,
            village TEXT,
            years_diabetes REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Scans table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            eye TEXT NOT NULL,
            grade INTEGER NOT NULL,
            grade_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            quality_score REAL NOT NULL,
            blur_score REAL NOT NULL,
            brightness_score REAL NOT NULL,
            image_path TEXT,
            heatmap_path TEXT,
            referral_advice TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

def add_patient(name: str, age: int, sex: str, phone: str = "", village: str = "", years_diabetes: float = 0.0) -> int:
    """Insert a new patient and return the generated ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients (name, age, sex, phone, village, years_diabetes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name.strip(), age, sex, phone.strip(), village.strip(), years_diabetes, datetime.now().isoformat()))
    patient_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return patient_id

def get_all_patients() -> List[Dict[str, Any]]:
    """Return all registered patients sorted by registration date."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_patient(patient_id: int) -> Optional[Dict[str, Any]]:
    """Get single patient details by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def search_patients(query: str) -> List[Dict[str, Any]]:
    """Search patients by name, village, or phone number."""
    if not query.strip():
        return get_all_patients()
    q = f"%{query.strip()}%"
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM patients
        WHERE name LIKE ? OR village LIKE ? OR phone LIKE ?
        ORDER BY id DESC
    """, (q, q, q))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def add_scan(
    patient_id: int,
    eye: str,
    grade: int,
    grade_name: str,
    confidence: float,
    quality_score: float,
    blur_score: float,
    brightness_score: float,
    image_path: str,
    heatmap_path: str,
    referral_advice: str
) -> int:
    """Save retinal screening scan results."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scans (
            patient_id, eye, grade, grade_name, confidence, quality_score,
            blur_score, brightness_score, image_path, heatmap_path, referral_advice, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id, eye, grade, grade_name, confidence, quality_score,
        blur_score, brightness_score, image_path, heatmap_path, referral_advice,
        datetime.now().isoformat()
    ))
    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def get_patient_scans(patient_id: int) -> List[Dict[str, Any]]:
    """Get historical scans for a given patient."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM scans
        WHERE patient_id = ?
        ORDER BY created_at DESC
    """, (patient_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

# Initialize database on import
init_db()
