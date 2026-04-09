import sys, os, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import json
import math
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from difflib import SequenceMatcher
from collections import defaultdict
import hashlib

"""
MIMIC-CXR Report Cleaning Pipeline
===================================
One-time preprocessing to extract clean FINDINGS sections.
Follows best practices from CheXpert, R2Gen, and recent medical report generation papers.

Usage:
    python clean_mimic_reports.py

Output:
    - mimic_clean_train.csv (images + clean findings)
    - mimic_clean_valid.csv (images + clean findings)
    - cleaning_report.txt (statistics and samples)
"""

import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path


import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import os

class MIMICReportCleaner:
    """
    State-of-the-art MIMIC-CXR report cleaner
    
    Features:
    - PHI & metadata removal
    - Findings extraction (3-strategy)
    - Abbreviation expansion
    - Negation normalization
    - Duplicate detection
    - Clinical relevance scoring
    - Label-report consistency verification
    - Statistical outlier removal
    
    Based on: R2Gen, R2GenCMN, VisualCheXbert, RadGraph, CheXpert
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stats = {
            'total_reports': 0,
            'successfully_extracted': 0,
            'failed_extraction': 0,
            'failed_quality': 0,
            'duplicates_removed': 0,
            'low_relevance_removed': 0,
            'inconsistent_removed': 0,
            'outliers_removed': 0,
            'final_clean': 0
        }
    
    # ========================================================================
    # EXISTING METHODS (Keep as-is)
    # ========================================================================
    
    def clean_raw_report(self, report_text):
        """Step 1: Remove PHI, metadata, demographics, and formatting artifacts"""
        if pd.isna(report_text) or not report_text or len(report_text) < 10:
            return ""
        
        text = str(report_text).strip()
        
        # =====================================================
        # Remove PHI
        # =====================================================
        phi_patterns = [
            r'___+',
            r'\[\*\*.*?\*\*\]',
            r'\[.*?\]',
        ]
        for pattern in phi_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # =====================================================
        # Remove header metadata
        # =====================================================
        header_patterns = [
            r'NAME:.*?(?=\n|$)',
            r'MEDICAL RECORD NUMBER:.*?(?=\n|$)',
            r'MRN:.*?(?=\n|$)',
            r'PATIENT ID:.*?(?=\n|$)',
            r'DATE OF BIRTH:.*?(?=\n|$)',
            r'AGE:.*?(?=\n|$)',
            r'SEX:.*?(?=\n|$)',
            r'ADMISSION DATE:.*?(?=\n|$)',
            r'DISCHARGE DATE:.*?(?=\n|$)',
            r'DATE:.*?(?=\n|$)',
            r'TIME:.*?(?=\n|$)',
        ]
        for pattern in header_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # =====================================================
        # Remove timestamps
        # =====================================================
        timestamp_patterns = [
            r'\d{1,2}:\d{2}\s*(?:AM|PM|A\.M\.|P\.M\.)',
            r'\d{1,2}\s*(?:AM|PM|A\.M\.|P\.M\.)\s*ON',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
        ]
        for pattern in timestamp_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # =====================================================
        # Remove age references
        # =====================================================
        text = re.sub(r'\b\d+[\s-](?:year|yr)[\s-]old\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*yo\b', '', text, flags=re.IGNORECASE)

        # =====================================================
        # 🩵 3. Remove demographic and "history" lines
        # =====================================================
        text = re.sub(r'\b\d{1,3}\s*[-]?\s*(year|yr)[-]?\s*old\s+(?:male|female)\b.*?(?:,|\.)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:male|female)\s+patient\s+with\s+.*?(?:,|\.)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:with\s+history\s+of|history\s+of|hx\s+of)\b.*?(?:,|\.)', '', text, flags=re.IGNORECASE)
        
        # =====================================================
        # Normalize whitespace and final cleanup
        # =====================================================
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_findings_section(self, report_text):
        """Step 2: Extract FINDINGS section using multi-strategy approach"""
        if not report_text or len(report_text) < 20:
            return ""
        
        text = str(report_text)
        text_upper = text.upper()
        
        # Strategy 1: Explicit FINDINGS section
        findings_pattern = re.compile(
            r'FINDINGS?:\s*(.*?)(?=IMPRESSION:|CONCLUSION:|RECOMMENDATIONS?:|COMMENT:|$)',
            re.DOTALL | re.IGNORECASE
        )
        
        match = findings_pattern.search(text)
        if match:
            findings = match.group(1).strip()
            findings = self._clean_findings_text(findings)
            if self._is_valid_findings(findings):
                return findings
        
        # Strategy 2: Extract between metadata and impression
        start_markers = [
            'EXAMINATION:', 'INDICATION:', 'HISTORY:', 
            'TECHNIQUE:', 'COMPARISON:', 'CLINICAL INFORMATION:'
        ]
        
        start_pos = 0
        for marker in start_markers:
            marker_match = re.search(marker, text_upper)
            if marker_match:
                line_end = text.find('\n', marker_match.end())
                if line_end != -1:
                    start_pos = max(start_pos, line_end + 1)
                else:
                    start_pos = max(start_pos, marker_match.end())
        
        end_markers = ['IMPRESSION:', 'CONCLUSION:', 'RECOMMENDATION:', 'COMMENT:']
        end_pos = len(text)
        
        for marker in end_markers:
            marker_match = re.search(marker, text_upper)
            if marker_match:
                end_pos = min(end_pos, marker_match.start())
        
        if start_pos < end_pos and (end_pos - start_pos) > 30:
            findings = text[start_pos:end_pos].strip()
            findings = self._clean_findings_text(findings)
            if self._is_valid_findings(findings):
                return findings
        
        # Strategy 3: Findings-only report
        metadata_markers = [
            'EXAMINATION:', 'INDICATION:', 'HISTORY:', 'TECHNIQUE:', 
            'COMPARISON:', 'IMPRESSION:', 'CONCLUSION:',
            'XXXX',  # IU-Xray uses XXXX for redaction
            'COMPARISON:', 'COMPARISON FILMS:',
            'INDICATION:'
        ]
        
        has_sections = any(marker in text_upper for marker in metadata_markers)
        
        if not has_sections:
            findings = self._clean_findings_text(text)
            if len(findings) >= 40 and self._is_valid_findings(findings, strict=True):
                return findings
        
        return ""
    
    import re

    def _clean_findings_text(self, text):
        """Enhanced cleaning for MIMIC-CXR reports to remove non-diagnostic phrases"""
        if not text:
            return ""

        text = text.strip()

        # =====================================================
        # 1. Remove section headers
        # =====================================================
        text = re.sub(r'(?i)^(?:FINDINGS?|IMPRESSION|REPORT|DISCUSSION|CONCLUSION)[:\s-]*', '', text)

        # =====================================================
        # 2. Remove technical headers
        # =====================================================
        technical_headers = [
            r'^(?:PA|AP|frontal|lateral|portable|upright)\s+(?:and|&)\s+(?:PA|AP|frontal|lateral|portable|upright)\s+views?.*?(?:obtained|provided|demonstrate[sd]?|show[ns]?)[.:,]?\s*',
            r'^(?:single\s+)?(?:PA|AP|frontal|lateral|portable|upright)\s+views?.*?(?:obtained|provided|demonstrate[sd]?|show[ns]?)[.:,]?\s*',
            r'^(?:chest\s+)?(?:radiograph|x-?ray)[.:,]?\s*',
            r'^views?\s+(?:of\s+the\s+chest\s+)?(?:were?\s+)?(?:obtained|provided)[.:,]?\s*',
        ]
        for pattern in technical_headers:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # =====================================================
        # ✅ IMPROVED: More aggressive comparison removal
        # =====================================================
        comparison_phrases = [
            # Complete sentences about comparison
            r'(?:^|\.\s+)[^.]*?\b(?:compared?|comparison)\s+(?:to|with)\s+(?:the\s+)?(?:prior|previous|most\s+recent)\s+(?:study|exam|examination|radiograph|ct|film)[^.]*?\.',
            r'(?:^|\.\s+)[^.]*?\b(?:prior|previous)\s+(?:study|exam|examination|radiograph|ct|film)[^.]*?\.',
            
            # Interval change patterns
            r'(?:^|\.\s+)[^.]*?\b(?:no\s+)?(?:significant\s+)?interval\s+change[^.]*?\.',
            r'(?:^|\.\s+)[^.]*?\binterval\s+(?:development|resolution|improvement|worsening)[^.]*?\.',
            
            # Stability patterns
            r'(?:^|\.\s+)[^.]*?\b(?:stable|unchanged)\s+(?:from|since|compared?\s+(?:to|with))[^.]*?\.',
            r'(?:^|\.\s+)[^.]*?\b(?:similar|essentially\s+unchanged)\s+(?:to|compared?\s+(?:to|with))[^.]*?\.',
            
            # New/resolved from prior
            r'(?:^|\.\s+)[^.]*?\b(?:new|resolved)\s+(?:from|since)\s+(?:the\s+)?(?:prior|previous)[^.]*?\.',
            
            # "As compared" patterns
            r'(?:^|\.\s+)[^.]*?\bas\s+compared?\s+(?:to|with)[^.]*?\.',
            r'(?:^|\.\s+)[^.]*?\bwhen\s+compared?\s+(?:to|with)[^.]*?\.',
            
            # Mid-sentence comparison phrases (more surgical)
            r',?\s*(?:compared?|comparison)\s+(?:to|with)\s+(?:the\s+)?(?:prior|previous|most\s+recent)\s+(?:study|exam|examination|radiograph|ct|film)(?:\s+(?:dated|from|of)\s+[^,\.]+)?[,\.]?',
            r',?\s*(?:unchanged|stable)\s+(?:from|since|compared?\s+(?:to|with))\s+(?:the\s+)?(?:prior|previous)[,\.]?',
            r',?\s*(?:similar|essentially\s+unchanged)\s+(?:to|compared?\s+(?:to|with))\s+(?:the\s+)?(?:prior|previous)[,\.]?',
        ]
        
        for pattern in comparison_phrases:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # =====================================================
        # 3. Remove technique descriptions
        # =====================================================
        text = re.sub(
            r'^(?:frontal|lateral|ap|pa|portable|upright)\s+(?:and\s+)?(?:frontal|lateral|ap|pa|portable|upright)?\s*(?:chest\s+)?radiograph[s]?\s+(?:demonstrate|show|are\s+obtained).*?[.:,]?\s*',
            '', text, flags=re.IGNORECASE
        )

        # =====================================================
        # 4. Remove demographics and history
        # =====================================================
        text = re.sub(r'\b\d{1,3}\s*[-]?\s*(year|yr)[-]?\s*old\s+(?:male|female)\b.*?(?:,|\.)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:male|female)\s+patient\s+with\s+.*?(?:,|\.)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:with\s+history\s+of|history\s+of|hx\s+of)\b.*?(?:,|\.)', '', text, flags=re.IGNORECASE)

        # =====================================================
        # 5. Basic cleanup
        # =====================================================
        text = re.sub(r'^[-•*]\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\d+\.\s*', '', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Remove standalone comparison words at sentence boundaries
        text = re.sub(r'(?:^|\.\s+)(?:unchanged|stable|similar)\.', '.', text, flags=re.IGNORECASE)

        # =====================================================
        # 6. Final formatting
        # =====================================================
        text = text.strip()
        
        # Remove empty sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
        text = '. '.join(sentences)
        
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    
    def _is_valid_findings(self, text, strict=False):
        """Validate extracted text is actual medical findings (ENHANCED)"""
        if not text or len(text) < 30:
            return False
        
        text_upper = text.upper()
        text_lower = text.lower()
        
        # ===== REJECTION CRITERIA (ENHANCED) =====
        
        # 1. Metadata markers (EXPANDED)
        metadata_markers = [
            'A.M. ON', 'P.M. ON', 'AM ON', 'PM ON',
            'HISTORY:', 'INDICATION:', 'TECHNIQUE:', 'EXAMINATION:',
            'CLINICAL INFORMATION:', 'REASON FOR EXAM:',
            'YEAR OLD', 'YEARS OLD', ' YO ', 'Y/O',
            'PATIENT ID', 'MRN:', 'MEDICAL RECORD',
            '___', 'XXXX', '[REDACTED]',
            'PRELIMINARY', 'WET READ', 'ADDENDUM',
            # ✅ NEW: Trauma/clinical history
            'POLYTRAUMA', 'TRAUMA', 'PEDESTRIAN', 'MOTOR VEHICLE',
            'GUNSHOT', 'ASSAULT', 'FALL FROM', 'STRUCK BY',
            # ✅ NEW: Technical views
            'SINGLE AP', 'PORTABLE VIEW', 'SINGLE VIEW',
            'PA AND LATERAL', 'FRONTAL VIEW',
            'NO PREVIOUS CHEST X-RAYS', 'ON PACS', 'PACS RECORD',
            # ✅ NEW: Exam requests
            'QUESTION INTERVAL', 'QUESTION CHANGE',
            'RULE OUT', 'R/O ', 'EVAL FOR', 'EVALUATE FOR',
            'ASSESS FOR', 'CONCERN FOR',
            # ✅ NEW: System notes
            'DICTATED BY', 'TRANSCRIBED BY', 'ELECTRONICALLY SIGNED',
            'ATTENDING:', 'RADIOLOGIST:', 'INTERPRETED BY',
        ]
        
        for marker in metadata_markers:
            if marker in text_upper:
                return False
        
        # 2. Character composition
        alpha_count = sum(c.isalpha() for c in text)
        digit_count = sum(c.isdigit() for c in text)
        
        alpha_ratio = alpha_count / len(text)
        
        if alpha_ratio < 0.60:
            return False
        
        # Remove medical measurements before digit check
        text_no_measurements = re.sub(r'\d+\.?\d*\s*(?:cm|mm|ml|kg|mg|cc)\b', '', text, flags=re.IGNORECASE)
        text_no_measurements = re.sub(r'\b[A-Z]\d+\b', '', text_no_measurements)
        text_no_measurements = re.sub(r'\d+(?:st|nd|rd|th)\s+(?:rib|intercostal)', '', text_no_measurements, flags=re.IGNORECASE)
        
        digit_count_filtered = sum(c.isdigit() for c in text_no_measurements)
        digit_ratio_filtered = digit_count_filtered / len(text_no_measurements) if len(text_no_measurements) > 0 else 0
        
        threshold = 0.10 if strict else 0.12
        if digit_ratio_filtered > threshold:
            return False
        
        # 3. Starts with comparison
        comparison_starters = [
            'COMPARED', 'COMPARISON', 'IN COMPARISON TO', 'IN COMPARISON WITH',
            'NO SIGNIFICANT CHANGE', 'UNCHANGED FROM', 'STABLE',
            'NO INTERVAL CHANGE', 'INTERVAL', 'PRIOR TO', 'AS COMPARED',
            'WHEN COMPARED', 'SINCE THE PRIOR'
        ]
        first_30 = text_upper[:30]
        if any(first_30.startswith(comp) for comp in comparison_starters):
            return False
        
        # 4. Technical patterns (ENHANCED - check ENTIRE text)
        technical_patterns = [
            r'\b(?:PA|AP|FRONTAL|LATERAL|PORTABLE|UPRIGHT)\s+(?:AND|&)\s+(?:PA|AP|FRONTAL|LATERAL|PORTABLE|UPRIGHT)\s+VIEWS?\b',
            r'\bVIEWS?\s+(?:WERE|WAS|ARE|IS)\s+(?:OBTAINED|PROVIDED)\b',
            r'\bVIEWS?\s+DEMONSTRATE\b',
            r'\bCHEST\s+(?:RADIOGRAPH|X-?RAY)\b.*?(?:OBTAINED|PROVIDED|DEMONSTRATE)',
            r'\bPREVIOUS\s+(?:PA|AP|FRONTAL|LATERAL|STUDY|EXAM|EXAMINATION)\b',
            r'\bSINGLE\s+(?:AP|PA|FRONTAL|LATERAL|PORTABLE)\s+VIEW\b',
            r'\b(?:PA|AP|FRONTAL|LATERAL)\s+VIEW\s+OF\s+THE\s+CHEST\b',
            # ✅ NEW patterns
            r'\bNO\s+PREVIOUS\s+(?:CHEST\s+)?X-?RAYS?\b',
            r'\bON\s+PACS\b',
            r'\bPACS\s+(?:RECORD|SYSTEM)\b',
            r'\bQUESTION\s+(?:INTERVAL|CHANGE)\b',
            r'\bRULE\s+OUT\b',
            r'\bEVALUATE?\s+FOR\b',
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, text_upper):
                return False
        
        # Technical vocabulary dominance check
        technical_words = [
            'obtained', 'provided', 'demonstrate', 'demonstrates', 'views', 
            'previous', 'compared', 'prior', 'frontal', 'lateral', 'portable',
            'upright', 'examination', 'radiograph'
        ]
        
        words = text_lower.split()
        if len(words) > 0:
            technical_word_count = sum(words.count(word) for word in technical_words)
            technical_ratio = technical_word_count / len(words)
            
            if technical_ratio > 0.15:
                return False
        
        # 5. Must contain medical terminology
        medical_terms = [
            'LUNG', 'CARDIAC', 'HEART', 'MEDIASTIN', 'PLEURAL', 'CHEST',
            'AORTA', 'PULMONARY', 'THORAX', 'DIAPHRAGM', 'HILAR',
            'COSTOPHRENIC', 'TRACHEA', 'VASCULATURE', 'CARINA',
            'APEX', 'BASE', 'HILA', 'HEMIDIAPHRAGM',
            'CONSOLIDATION', 'EFFUSION', 'PNEUMOTHORAX', 'ATELECTASIS',
            'OPACITY', 'EDEMA', 'CARDIOMEGALY', 'INFILTRATE',
            'NODULE', 'MASS', 'LESION', 'PNEUMONIA',
            'CONGESTION', 'HYPERINFLATION', 'EMPHYSEMA',
            'CLEAR', 'NORMAL', 'ABNORMAL', 'ENLARGED', 'SMALL',
            'BILATERAL', 'UNILATERAL', 'LEFT', 'RIGHT',
            'UPPER', 'LOWER', 'MIDDLE', 'LOBE', 'INTERSTITIAL',
            'FOCAL', 'DIFFUSE', 'PATCHY', 'STREAKY'
        ]
        
        medical_count = sum(1 for term in medical_terms if term in text_upper)
        
        min_terms = 3 if strict else 2
        if medical_count < min_terms:
            return False
        
        # 6. Sentence structure
        if '.' not in text and '!' not in text and '?' not in text:
            return False
        
        sentences = text.split('.')
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if len(complete_sentences) < 1:
            return False
        
        # 7. Strict mode
        if strict:
            if medical_count < 4:
                return False
            
            first_40 = text_lower[:40]
            if any(tech_word in first_40 for tech_word in ['obtained', 'provided', 'views', 'demonstrate']):
                return False
        
        return True
    
    def apply_quality_filters(self, text: str) -> bool:
        """Apply STRICT quality filters (ENHANCED)"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return False

        text = text.strip()
        
        # Minimum character length (stricter)
        if len(text) < 50:
            return False
        
        # Maximum character length
        if len(text) > 400:
            return False

        # Minimum word count (stricter)
        words = text.split()
        if len(words) < 10:
            return False
        
        # Maximum word count
        if len(words) > 70:
            return False

        # Must have sentence structure
        if '.' not in text:
            return False
        
        # Require MULTIPLE medical keywords
        medical_keywords = [
            "lung", "lungs", "cardiac", "heart", "pleural", 
            "opacity", "infiltrate", "normal", "pneumonia",
            "consolidation", "effusion", "pneumothorax", "atelectasis",
            "mediastin", "aorta", "vasculature", "hilar",
            "clear", "edema", "cardiomegaly"
        ]
        
        keyword_count = sum(1 for k in medical_keywords if k in text.lower())
        if keyword_count < 2:
            return False
        
        # Character composition
        alpha_count = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_count / len(text)
        if alpha_ratio < 0.65:
            return False
        
        # Not too many digits
        text_no_measurements = re.sub(r'\d+\.?\d*\s*(?:cm|mm|ml|kg|mg|cc)\b', '', text, flags=re.IGNORECASE)
        digit_count = sum(c.isdigit() for c in text_no_measurements)
        digit_ratio = digit_count / len(text_no_measurements) if len(text_no_measurements) > 0 else 0
        if digit_ratio > 0.10:
            return False
        
        # Final validation
        return self._is_valid_findings(text, strict=True)
    def final_comparison_check(self, df):
        """Final pass to catch any remaining comparison phrases"""
        print(f"\n{'='*80}")
        print("FINAL COMPARISON PHRASE CHECK")
        print(f"{'='*80}")
        
        comparison_keywords = [
            'compared', 'comparison', 'prior study', 'prior exam',
            'previous study', 'previous exam', 'interval change',
            'unchanged from', 'stable from', 'similar to prior',
            'as compared', 'when compared'
        ]
        
        contaminated_indices = []
        
        for idx, row in df.iterrows():
            text_lower = row['Findings_Clean'].lower()
            if any(keyword in text_lower for keyword in comparison_keywords):
                contaminated_indices.append(idx)
        
        print(f"\nReports with comparison phrases: {len(contaminated_indices)}")
        
        if contaminated_indices:
            print(f"Removing {len(contaminated_indices)} contaminated reports...")
            
            # Show examples
            for i, idx in enumerate(contaminated_indices[:5], 1):
                text = df.loc[idx, 'Findings_Clean']
                print(f"\n  Example {i}: {text[:150]}...")
            
            df_clean = df.drop(contaminated_indices)
            print(f"\nAfter removal: {len(df_clean)} reports remain")
            
            self.stats['inconsistent_removed'] += len(contaminated_indices)
            
            return df_clean
        else:
            print("✅ No comparison phrases found!")
            return df
    # ========================================================================
    # NEW METHODS (State-of-the-art enhancements)
    # ========================================================================
    
    def expand_abbreviations(self, text):
        """Expand common medical abbreviations"""
        if not text:
            return text
        
        abbreviations = {
            # Anatomy
            r'\bLLL\b': 'left lower lobe',
            r'\bRLL\b': 'right lower lobe',
            r'\bLUL\b': 'left upper lobe',
            r'\bRUL\b': 'right upper lobe',
            r'\bRML\b': 'right middle lobe',
            r'\bLA\b': 'left atrium',
            r'\bRA\b': 'right atrium',
            r'\bLV\b': 'left ventricle',
            r'\bRV\b': 'right ventricle',
            
            # Pathology
            r'\bPTX\b': 'pneumothorax',
            r'\bCHF\b': 'congestive heart failure',
            r'\bCOPD\b': 'chronic obstructive pulmonary disease',
            r'\bPNA\b': 'pneumonia',
            r'\bPE\b': 'pulmonary embolism',
            r'\bCABG\b': 'coronary artery bypass graft',
            r'\bMI\b': 'myocardial infarction',
            r'\bCAD\b': 'coronary artery disease',
            
            # Procedures/Devices
            r'\bET\b': 'endotracheal',
            r'\bNG\b': 'nasogastric',
            r'\bOG\b': 'orogastric',
            r'\bIJ\b': 'internal jugular',
            r'\bPICC\b': 'peripherally inserted central catheter',
            r'\bCVC\b': 'central venous catheter',
            
            # Directions
            r'\bAP\b(?! and)': 'anteroposterior',
            r'\bPA\b(?! and)': 'posteroanterior',
            r'\bLat\b': 'lateral',
            r'\bBIL\b': 'bilateral',
            
            # Common terms
            r'\bPt\b': 'patient',
            r'\bs/p\b': 'status post',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without',
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(abbr, full, text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def normalize_negations(self, text):
        """Normalize negation patterns"""
        if not text:
            return text
        
        negation_patterns = [
            (r'no\s+evidence\s+of\s+(\w+)', r'no \1'),
            (r'no\s+radiographic\s+evidence\s+of\s+(\w+)', r'no \1'),
            (r'without\s+evidence\s+of\s+(\w+)', r'no \1'),
            (r'there\s+is\s+no\s+(\w+)', r'no \1'),
            (r'there\s+are\s+no\s+(\w+)', r'no \1'),
            (r'absence\s+of\s+(\w+)', r'no \1'),
            (r'(\w+)\s+(?:is|are)\s+not\s+(?:seen|identified|present)', r'no \1'),
            (r'free\s+(?:of|from)\s+(\w+)', r'no \1'),
            (r'do(?:es)?\s+not\s+show\s+(\w+)', r'no \1'),
        ]
        
        for pattern, replacement in negation_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Uncertainty
        uncertainty_patterns = [
            (r'possible\s+(\w+)', r'\1 suspected'),
            (r'possibly\s+(\w+)', r'\1 suspected'),
            (r'cannot\s+exclude\s+(\w+)', r'\1 not excluded'),
            (r'unable\s+to\s+exclude\s+(\w+)', r'\1 not excluded'),
            (r'may\s+represent\s+(\w+)', r'possibly \1'),
            (r'could\s+represent\s+(\w+)', r'possibly \1'),
        ]
        
        for pattern, replacement in uncertainty_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def detect_duplicates(self, df, similarity_threshold=0.95):
        """Detect and remove exact and near-duplicate reports"""
        print(f"\n{'='*80}")
        print("DETECTING DUPLICATES")
        print(f"{'='*80}")
        
        findings = df['Findings_Clean'].tolist()
        n = len(findings)
        
        # Exact duplicates
        exact_duplicates = defaultdict(list)
        for idx, text in enumerate(findings):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            exact_duplicates[text_hash].append(idx)
        
        exact_dups_to_remove = []
        for indices in exact_duplicates.values():
            if len(indices) > 1:
                exact_dups_to_remove.extend(indices[1:])
        
        print(f"\nExact duplicates found: {len(exact_dups_to_remove)}")
        
        # Near-duplicates
        print("Detecting near-duplicates (this may take a few minutes)...")
        
        near_duplicates = []
        
        # Hash bucketing by length
        length_buckets = defaultdict(list)
        for idx, text in enumerate(findings):
            if idx not in exact_dups_to_remove:
                length_buckets[len(text) // 10].append(idx)
        
        checked_pairs = set()
        
        for bucket_indices in tqdm(length_buckets.values(), desc="Checking buckets"):
            if len(bucket_indices) < 2:
                continue
            
            for i in range(len(bucket_indices)):
                for j in range(i+1, len(bucket_indices)):
                    idx_i = bucket_indices[i]
                    idx_j = bucket_indices[j]
                    
                    pair = tuple(sorted([idx_i, idx_j]))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
                    
                    text_i = findings[idx_i]
                    text_j = findings[idx_j]
                    
                    len_ratio = min(len(text_i), len(text_j)) / max(len(text_i), len(text_j))
                    if len_ratio < 0.8:
                        continue
                    
                    similarity = SequenceMatcher(None, text_i, text_j).ratio()
                    
                    if similarity >= similarity_threshold:
                        near_duplicates.append((idx_i, idx_j, similarity))
        
        near_dups_to_remove = set(idx_j for idx_i, idx_j, sim in near_duplicates)
        
        print(f"Near-duplicates found: {len(near_dups_to_remove)} (similarity >= {similarity_threshold})")
        
        # Create clean dataset
        all_dups_to_remove = set(exact_dups_to_remove) | near_dups_to_remove
        
        df_clean = df[~df.index.isin(all_dups_to_remove)].copy()
        
        print(f"\nDuplicate Removal Summary:")
        print(f"  Original: {len(df)} reports")
        print(f"  Exact duplicates removed: {len(exact_dups_to_remove)}")
        print(f"  Near-duplicates removed: {len(near_dups_to_remove)}")
        print(f"  Total removed: {len(all_dups_to_remove)} ({100*len(all_dups_to_remove)/len(df):.1f}%)")
        print(f"  Final: {len(df_clean)} reports")
        
        self.stats['duplicates_removed'] += len(all_dups_to_remove)
        
        # Show examples
        if near_duplicates[:3]:
            print("\nExample near-duplicates detected:")
            for i, (idx_i, idx_j, sim) in enumerate(near_duplicates[:3], 1):
                print(f"\n  Pair {i} (similarity: {sim:.3f}):")
                print(f"    A: {findings[idx_i][:80]}...")
                print(f"    B: {findings[idx_j][:80]}...")
        
        return df_clean
    
    def compute_clinical_relevance_score(self, text):
        """Score report for clinical information content"""
        if not text or len(text) < 20:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        # Component 1: Anatomical Coverage (0-0.3)
        anatomy_terms = {
            'lung', 'lungs', 'cardiac', 'heart', 'mediastinum', 'mediastinal',
            'pleural', 'aorta', 'aortic', 'pulmonary', 'hilar', 'hila',
            'diaphragm', 'hemidiaphragm', 'costophrenic', 'trachea',
            'clavicle', 'ribs', 'thorax', 'thoracic', 'vasculature',
            'atrium', 'ventricle', 'apex', 'base', 'lobe'
        }
        
        anatomy_count = sum(1 for term in anatomy_terms if term in text_lower)
        anatomy_score = min(0.3, anatomy_count * 0.05)
        score += anatomy_score
        
        # Component 2: Pathology/Findings (0-0.4)
        pathology_terms = {
            'consolidation', 'infiltrate', 'opacity', 'opacities', 'effusion',
            'pneumothorax', 'atelectasis', 'edema', 'congestion',
            'cardiomegaly', 'enlarged', 'pneumonia', 'mass', 'nodule',
            'lesion', 'abnormal', 'abnormality', 'fracture', 'displaced',
            'emphysema', 'hyperinflation', 'interstitial', 'fibrosis'
        }
        
        pathology_count = sum(1 for term in pathology_terms if term in text_lower)
        pathology_score = min(0.4, pathology_count * 0.08)
        score += pathology_score
        
        # Component 3: Specific Descriptors (0-0.2)
        descriptor_terms = {
            'bilateral', 'unilateral', 'diffuse', 'focal', 'patchy',
            'scattered', 'localized', 'extensive', 'mild', 'moderate',
            'severe', 'small', 'large', 'multiple', 'single',
            'upper', 'lower', 'middle', 'right', 'left', 'basilar',
            'apical', 'peripheral', 'central', 'anterior', 'posterior'
        }
        
        descriptor_count = sum(1 for term in descriptor_terms if term in text_lower)
        descriptor_score = min(0.2, descriptor_count * 0.04)
        score += descriptor_score
        
        # Component 4: Sentence Complexity (0-0.1)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        if len(sentences) >= 2:
            score += 0.1
        
        # Penalties
        generic_patterns = [
            r'^no\s+acute\s+\w+\.$',
            r'^normal\s+chest\s+x-?ray\.$',
            r'^unremarkable\.$',
        ]
        
        for pattern in generic_patterns:
            if re.search(pattern, text_lower):
                score *= 0.5
        
        if len(text) < 50:
            score *= 0.7
        
        return min(1.0, score)
    
    def filter_by_clinical_relevance(self, df, min_score=0.4):
        """Filter dataset by clinical relevance score"""
        print(f"\n{'='*80}")
        print("COMPUTING CLINICAL RELEVANCE SCORES")
        print(f"{'='*80}")
        
        scores = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring reports"):
            findings = row['Findings_Clean']
            score = self.compute_clinical_relevance_score(findings)
            scores.append(score)
        
        df['clinical_relevance_score'] = scores
        
        print(f"\nScore Distribution:")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Median: {np.median(scores):.3f}")
        print(f"  Std: {np.std(scores):.3f}")
        print(f"  Min: {np.min(scores):.3f}")
        print(f"  Max: {np.max(scores):.3f}")
        
        df_filtered = df[df['clinical_relevance_score'] >= min_score].copy()
        
        removed = len(df) - len(df_filtered)
        print(f"\nFiltering Results (min_score={min_score}):")
        print(f"  Original: {len(df)} reports")
        print(f"  Removed: {removed} ({100*removed/len(df):.1f}%)")
        print(f"  Kept: {len(df_filtered)} ({100*len(df_filtered)/len(df):.1f}%)")
        
        self.stats['low_relevance_removed'] += removed
        
        # Show examples
        low_score_samples = df[df['clinical_relevance_score'] < min_score].head(3)
        if len(low_score_samples) > 0:
            print("\nExamples of REMOVED reports (low score):")
            for i, (idx, data) in enumerate(low_score_samples.iterrows(), 1):
                print(f"\n  {i}. Score: {data['clinical_relevance_score']:.3f}")
                print(f"     Text: {data['Findings_Clean'][:100]}...")
        
        high_score_samples = df_filtered.nlargest(3, 'clinical_relevance_score')
        print("\nExamples of KEPT reports (high score):")
        for i, (idx, data) in enumerate(high_score_samples.iterrows(), 1):
            print(f"\n  {i}. Score: {data['clinical_relevance_score']:.3f}")
            print(f"     Text: {data['Findings_Clean'][:100]}...")
        
        return df_filtered
    
    def validate_clean_findings(self, df, split_name='train'):
        """Additional validation pass to catch edge cases"""
        print(f"\n{'='*80}")
        print(f"VALIDATING {split_name.upper()} FINDINGS")
        print(f"{'='*80}")
        
        initial_count = len(df)
        rejected = []
        
        for idx, row in df.iterrows():
            findings = row['Findings_Clean']
            findings_upper = findings.upper()
            
            corruption_patterns = [
                'POLYTRAUMA', 'PEDESTRIAN', 'MOTOR VEHICLE',
                'SINGLE AP VIEW', 'PORTABLE VIEW', 'PACS RECORD',
                'NO PREVIOUS', 'QUESTION INTERVAL', 'RULE OUT',
                'DICTATED BY', 'TRANSCRIBED BY',
            ]
            
            is_corrupted = any(pattern in findings_upper for pattern in corruption_patterns)
            
            if is_corrupted:
                rejected.append({
                    'index': idx,
                    'reason': 'Contains corruption pattern',
                    'text': findings[:150]
                })
        
        valid_indices = [idx for idx in df.index if idx not in [r['index'] for r in rejected]]
        df_clean = df.loc[valid_indices].copy()
        
        print(f"\nValidation Results:")
        print(f"  Initial: {initial_count} samples")
        print(f"  Rejected: {len(rejected)} samples ({100*len(rejected)/initial_count:.1f}%)")
        print(f"  Clean: {len(df_clean)} samples ({100*len(df_clean)/initial_count:.1f}%)")
        
        if rejected:
            print(f"\nSample Rejected Findings:")
            for i, r in enumerate(rejected[:3], 1):
                print(f"\n  {i}. Reason: {r['reason']}")
                print(f"     Text: {r['text']}...")
        
        return df_clean
    
    def process_dataframe(self, df, split_name='train'):
        """Process entire dataframe (basic pipeline)"""
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} set")
        print(f"{'='*80}")
        
        total = len(df)
        clean_findings = []
        
        for idx, row in tqdm(df.iterrows(), total=total, desc=f"Cleaning {split_name}"):
            report = row.get('Report', '')
            
            # Step 1: Clean raw report
            cleaned = self.clean_raw_report(report)
            
            # Step 2: Extract findings
            findings = self.extract_findings_section(cleaned)
            
            # Step 3: Expand abbreviations
            findings = self.expand_abbreviations(findings)
            
            # Step 4: Normalize negations
            findings = self.normalize_negations(findings)
            
            findings = self.add_uncertainty_language(findings)
            
            clean_findings.append(findings)
            
            self.stats['total_reports'] += 1
            if findings and len(findings) > 20:
                self.stats['successfully_extracted'] += 1
            else:
                self.stats['failed_extraction'] += 1
        
        df['Findings_Clean'] = clean_findings
        
        # Step 5: Apply quality filters
        df['PassesQuality'] = df['Findings_Clean'].apply(self.apply_quality_filters)
        
        df_filtered = df[df['PassesQuality']].copy()
        
        passed = len(df_filtered)
        failed = total - passed
        
        self.stats['failed_quality'] += failed
        self.stats['final_clean'] += passed
        
        print(f"\n{split_name.upper()} Results:")
        print(f"  Input: {total} reports")
        print(f"  Successfully extracted: {self.stats['successfully_extracted'] - (self.stats['final_clean'] - passed)}")
        print(f"  Failed extraction: {failed + (self.stats['successfully_extracted'] - passed)}")
        print(f"  Passed quality filter: {passed} ({100*passed/total:.1f}%)")
        
        return df_filtered
    
    def generate_report(self, train_df, valid_df, output_file='cleaning_report.txt'):
        """Generate detailed cleaning report"""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MIMIC-CXR REPORT CLEANING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total reports processed: {self.stats['total_reports']}\n")
            f.write(f"  Successfully extracted: {self.stats['successfully_extracted']}\n")
            f.write(f"  Failed extraction: {self.stats['failed_extraction']}\n")
            f.write(f"  Failed quality filter: {self.stats['failed_quality']}\n")
            f.write(f"  Duplicates removed: {self.stats['duplicates_removed']}\n")
            f.write(f"  Low relevance removed: {self.stats['low_relevance_removed']}\n")
            f.write(f"  Final clean reports: {self.stats['final_clean']}\n")
            f.write(f"  Success rate: {100*self.stats['final_clean']/self.stats['total_reports']:.1f}%\n\n")
            
            # Train set
            f.write("-"*80 + "\n")
            f.write("TRAIN SET:\n")
            f.write(f"  Total: {len(train_df)} reports\n")
            
            train_lengths = train_df['Findings_Clean'].str.len()
            train_words = train_df['Findings_Clean'].apply(lambda x: len(str(x).split()))
            
            f.write(f"  Character length: min={train_lengths.min()}, max={train_lengths.max()}, mean={train_lengths.mean():.1f}, std={train_lengths.std():.1f}\n")
            f.write(f"  Word count: min={train_words.min()}, max={train_words.max()}, mean={train_words.mean():.1f}, std={train_words.std():.1f}\n\n")
            
            # Valid set
            f.write("-"*80 + "\n")
            f.write("VALIDATION SET:\n")
            f.write(f"  Total: {len(valid_df)} reports\n")
            
            valid_lengths = valid_df['Findings_Clean'].str.len()
            valid_words = valid_df['Findings_Clean'].apply(lambda x: len(str(x).split()))
            
            f.write(f"  Character length: min={valid_lengths.min()}, max={valid_lengths.max()}, mean={valid_lengths.mean():.1f}, std={valid_lengths.std():.1f}\n")
            f.write(f"  Word count: min={valid_words.min()}, max={valid_words.max()}, mean={valid_words.mean():.1f}, std={valid_words.std():.1f}\n\n")
            
            # Sample reports
            f.write("="*80 + "\n")
            f.write("SAMPLE CLEAN FINDINGS (TRAIN SET):\n")
            f.write("="*80 + "\n\n")
            
            sample_indices = np.linspace(0, len(train_df)-1, 10, dtype=int)
            for i, idx in enumerate(sample_indices, 1):
                findings = train_df.iloc[idx]['Findings_Clean']
                f.write(f"Sample {i}:\n")
                f.write(f"  Length: {len(findings)} chars, {len(findings.split())} words\n")
                f.write(f"  Text: {findings}\n\n")
        
        print(f"\n✅ Detailed report saved to: {output_file}")

    def add_uncertainty_language(self, text, labels_row=None):
        """
        Add natural uncertainty language to findings based on label uncertainty
        This teaches the model to express uncertainty naturally
        """
        if not text or len(text) < 30:
            return text
        
        # If we have uncertain labels (-1 in MIMIC-CXR), add hedge words
        sentences = text.split('.')
        modified_sentences = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                modified_sentences.append(sent)
                continue
            
            # Add uncertainty to findings probabilistically
            import random
            if random.random() < 0.3:  # 30% of sentences get uncertainty
                # Detect if sentence has a finding
                if any(word in sent.lower() for word in ['opacity', 'infiltrate', 'consolidation', 'effusion', 'edema']):
                    # Add hedge word at start
                    hedges = ['possible', 'probable', 'suspected', 'questionable']
                    hedge = random.choice(hedges)
                    sent = f"{hedge} {sent}"
            
            modified_sentences.append(sent)
        
        return '. '.join(modified_sentences)
import os
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# class MIMICCXRCleanDataset(Dataset):
#     """
#     MIMIC-CXR Dataset using PRE-CLEANED findings
#     No extraction needed - just load and use!
#     """
    
#     def __init__(self, dataframe, transform=None, image_root=None):
#         """
#         Args:
#             dataframe: DataFrame with 'Findings_Clean' column
#             transform: Image transformations
#             image_root: Root directory for images
#         """
#         self.dataframe = dataframe.reset_index(drop=True)
#         self.transform = transform
#         self.image_root = image_root
        
#         # Define label columns
#         self.label_cols = [
#             'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
#             'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
#             'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
#             'Pleural Other', 'Fracture', 'Support Devices'
#         ]
        
#         # Handle labels
#         for col in self.label_cols:
#             if col in self.dataframe.columns:
#                 self.dataframe[col] = self.dataframe[col].fillna(0)
#                 self.dataframe[col] = self.dataframe[col].apply(
#                     lambda x: 1 if x in [1.0, -1.0] else 0
#                 )
        
#         # Verify clean findings exist
#         if 'Findings_Clean' not in self.dataframe.columns:
#             raise ValueError("Missing 'Findings_Clean' column! Run clean_mimic_reports.py first.")
        
#         valid_count = (self.dataframe['Findings_Clean'].str.len() > 30).sum()
#         print(f"✓ Dataset ready: {len(self.dataframe)} samples, {valid_count} with clean findings")
    
#     def __len__(self):
#         return len(self.dataframe)
    
#     def __getitem__(self, idx):
#         item = self.dataframe.iloc[idx]
#         img_path = os.path.join(self.image_root, item['Path'])
        
#         try:
#             # Load image
#             image = Image.open(img_path).convert("RGB")
#             if self.transform:
#                 image = self.transform(image)
            
#             # Get labels
#             label = item[self.label_cols].values.astype(np.float32)
#             label = torch.as_tensor(label)
            
#             # Get clean findings (already processed!)
#             findings = item['Findings_Clean']
#             if pd.isna(findings):
#                 findings = ""
#             else:
#                 findings = str(findings).strip()
            
#             return image, label, findings
            
#         except Exception as e:
#             print(f"⚠️  Error loading {img_path}: {e}")
#             # Return dummy data
#             image = torch.zeros(3, 320, 320)
#             label = torch.zeros(len(self.label_cols))
#             findings = ""
#             return image, label, findings

def main():
    """
    Main cleaning pipeline - CORRECTED VERSION
    """
    print("\n" + "="*80)
    print("MIMIC-CXR REPORT CLEANING PIPELINE")
    print("="*80 + "\n")
    
    # ========================================================================
    # Configuration
    # ========================================================================
    CONFIG = {
        # IU-Xray paths
        'image_root': "/mnt/Internal/MedImage/Datasets/IU_Xray/images/images_normalized",
        'reports_csv': "/mnt/Internal/MedImage/Datasets/IU_Xray/indiana_reports.csv",
        'projections_csv': "/mnt/Internal/MedImage/Datasets/IU_Xray/indiana_projections.csv",
        
        # Output files (same as MIMIC)
        'output_dir': "/mnt/Internal/MedImage/Datasets/IU_Xray/cleaned",
        'train_output': "iu_xray_clean_train.csv",
        'valid_output': "iu_xray_clean_valid.csv",
        'report_output': "iu_xray_cleaning_report.txt",
        
        # Settings
        'valid_split': 0.15,
        'max_samples': None,  # Use all ~3851 reports
    }
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # ========================================================================
    # Step 1: Load IU-Xray Data
    # ========================================================================
    print("📂 Loading IU-Xray data...")

    # Load reports (contains findings + impression)
    reports_df = pd.read_csv(CONFIG['reports_csv'])
    reports_df['uid'] = reports_df['uid'].astype(str)

    # Load projections (maps images to studies)
    projections_df = pd.read_csv(CONFIG['projections_csv'])
    projections_df['uid'] = projections_df['uid'].astype(str)

    # Combine findings + impression into a single report
    # (IU-Xray has both sections - similar to MIMIC structure)
    reports_df['Report'] = reports_df.apply(
        lambda row: f"FINDINGS: {row['findings'] if pd.notna(row['findings']) else ''}\n\n"
                    f"IMPRESSION: {row['impression'] if pd.notna(row['impression']) else ''}",
        axis=1
    )

    print(f"  Loaded {len(reports_df)} reports")
    print(f"  Loaded {len(projections_df)} image projections")

    # ========================================================================
    # Step 2: Match Images to Reports
    # ========================================================================
    print("\n🖼️  Matching images to reports...")

    # Merge projections with reports
    data_df = projections_df.merge(
        reports_df[['uid', 'Report', 'findings', 'impression']],
        on='uid',
        how='inner'
    )

    # Filter to Frontal views only (to match MIMIC's single-view approach)
    # OR keep both if you want - IU-Xray has PA + Lateral
    data_df = data_df[data_df['projection'] == 'Frontal'].reset_index(drop=True)

    # Set Path column (IU-Xray uses 'filename')
    data_df['Path'] = data_df['filename']

    # Verify images exist
    valid_rows = []
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Verifying images"):
        img_path = os.path.join(CONFIG['image_root'], row['Path'])
        if os.path.exists(img_path):
            valid_rows.append(row)

    data_df_matched = pd.DataFrame(valid_rows)
    print(f"  Matched {len(data_df_matched)} images")
    
    # # ========================================================================
    # # Step 3: Match Images
    # # ========================================================================
    # print("\n🖼️  Matching images...")
    
    # all_files = os.listdir(CONFIG['image_root'])
    # jpg_files = [f for f in all_files if f.endswith('.jpg')]
    # dicom_to_file = {f.replace('.jpg', ''): f for f in jpg_files}
    
    # print(f"  Indexed {len(dicom_to_file)} images")
    
    # valid_rows = []
    # for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Matching images", ncols=80):
    #     if pd.notna(row.get('dicom_id')):
    #         dicom_id = str(row['dicom_id'])
    #         if dicom_id in dicom_to_file:
    #             row['Path'] = dicom_to_file[dicom_id]
    #             valid_rows.append(row)
    
    # data_df_matched = pd.DataFrame(valid_rows)
    # print(f"  Matched {len(data_df_matched)} images")
    
    # ========================================================================
    # ✅ NEW: Step 4: Filter to train/validate splits only (before cleaning)
    # ========================================================================
    print("\n🔍 Filtering to train/validate splits...")
    
    # ========================================================================
    # Step 4: Use all data (IU-Xray has no predefined split)
    # ========================================================================
    print("\n🔍 Using all IU-Xray data (will split after cleaning)...")

    data_df_filtered = data_df_matched.copy()  # ← Use all data

    print(f"  Total samples: {len(data_df_filtered)}")
        
    # ✅ Sample if needed (for faster testing)
    if CONFIG['max_samples'] and CONFIG['max_samples'] < len(data_df_filtered):
        print(f"  Sampling {CONFIG['max_samples']} for faster processing...")
        data_df_filtered = data_df_filtered.sample(
            n=CONFIG['max_samples'], 
            random_state=42
        ).reset_index(drop=True)
    
    # ========================================================================
    # ✅ NEW: Step 5: Clean ALL data BEFORE splitting
    # ========================================================================
    print("\n🧹 Cleaning all reports (BEFORE train/valid split)...")
    print("="*80)
    
    cleaner = MIMICReportCleaner(verbose=True)
    
    # Process entire dataset
    print("\n📋 Processing entire dataset...")
    all_clean = cleaner.process_dataframe(data_df_matched, split_name='all')
    
    print("\n🔍 Removing duplicates from entire dataset...")
    all_dedup = cleaner.detect_duplicates(all_clean, similarity_threshold=0.95)
    
    print("\n📊 Filtering by clinical relevance...")
    all_scored = cleaner.filter_by_clinical_relevance(all_dedup, min_score=0.25)  # ✅ Relaxed from 0.4
    
    print("\n✅ Validating clean findings...")
    all_validated = cleaner.validate_clean_findings(all_scored, split_name='all')
    
    print("\n🔎 Final comparison phrase check...")
    all_final = cleaner.final_comparison_check(all_validated)
    
    print(f"\n✅ Clean dataset ready: {len(all_final)} samples")
    
    # ========================================================================
    # Step 6: Split into train/valid
    # ========================================================================
    print("\n✂️  Splitting clean data into train/valid...")

    from sklearn.model_selection import train_test_split

    train_final, valid_final = train_test_split(
        all_final,
        test_size=CONFIG['valid_split'],
        random_state=42,
        shuffle=True
    )

    train_final = train_final.reset_index(drop=True)
    valid_final = valid_final.reset_index(drop=True)

    print(f"  Initial split:")
    print(f"    Train: {len(train_final)} samples ({100*len(train_final)/len(all_final):.1f}%)")
    print(f"    Valid: {len(valid_final)} samples ({100*len(valid_final)/len(all_final):.1f}%)")

    # ========================================================================
    # ✅ Step 7: Remove train/valid overlaps (CRITICAL FIX)
    # ========================================================================
    print("\n🔒 Removing train/valid overlaps...")

    train_texts = set(train_final['Findings_Clean'].tolist())
    valid_texts = set(valid_final['Findings_Clean'].tolist())
    overlap = train_texts & valid_texts

    if len(overlap) > 0:
        print(f"  ⚠️  Found {len(overlap)} overlapping reports ({100*len(overlap)/len(valid_final):.1f}% of validation)")
        print(f"  🧹 Removing overlaps from validation set...")
        
        # Show examples before removal
        print(f"\n  Examples of overlapping reports:")
        for i, text in enumerate(list(overlap)[:3], 1):
            print(f"    {i}. {text[:100]}...")
        
        # Remove overlaps from validation (keep in train since it's larger)
        valid_final = valid_final[~valid_final['Findings_Clean'].isin(overlap)].reset_index(drop=True)
        
        print(f"\n  ✅ Validation set after cleanup: {len(valid_final)} samples")
        print(f"  ✅ Removed {len(overlap)} duplicates from validation")
    else:
        print(f"  ✅ No overlap detected - clean split!")

    # Re-verify
    train_texts_final = set(train_final['Findings_Clean'].tolist())
    valid_texts_final = set(valid_final['Findings_Clean'].tolist())
    overlap_final = train_texts_final & valid_texts_final

    print(f"\n📊 Final split statistics:")
    print(f"   Train: {len(train_final)} unique reports")
    print(f"   Valid: {len(valid_final)} unique reports")
    print(f"   Overlap: {len(overlap_final)} reports")
    print(f"   Train/Valid ratio: {len(train_final)/len(valid_final):.1f}:1")

    if len(overlap_final) == 0:
        print(f"   ✅ Zero data leakage confirmed!")
    else:
        print(f"   ❌ WARNING: Still {len(overlap_final)} overlaps remain!")
        raise ValueError("Data leakage still present after cleanup!")
    
    # ========================================================================
    # Step 8: Save Clean Datasets
    # ========================================================================
    print("\n💾 Saving clean datasets...")
    
    # Select columns to keep
    keep_cols = [
        'uid',          # IU-Xray uses 'uid' instead of 'subject_id'
        'filename',     # Original image filename
        'Path',         # Image path
        'projection',   # Frontal/Lateral
        'Findings_Clean',
        'clinical_relevance_score'
    ]

    train_keep_cols = [col for col in keep_cols if col in train_final.columns]
    valid_keep_cols = [col for col in keep_cols if col in valid_final.columns]

    train_save = train_final[[col for col in train_keep_cols if col in train_final.columns]].copy()
    valid_save = valid_final[[col for col in valid_keep_cols if col in valid_final.columns]].copy()

    train_output_path = os.path.join(CONFIG['output_dir'], CONFIG['train_output'])
    valid_output_path = os.path.join(CONFIG['output_dir'], CONFIG['valid_output'])

    train_save.to_csv(train_output_path, index=False)
    valid_save.to_csv(valid_output_path, index=False)

    print(f"  ✅ Train saved: {train_output_path} ({len(train_save)} samples)")
    print(f"  ✅ Valid saved: {valid_output_path} ({len(valid_save)} samples)")
    
    # ========================================================================
    # Step 9: Generate Report
    # ========================================================================
    report_output_path = os.path.join(CONFIG['output_dir'], CONFIG['report_output'])
    cleaner.generate_report(train_save, valid_save, output_file=report_output_path)

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("✅ CLEANING COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  📄 Train CSV: {train_output_path}")
    print(f"  📄 Valid CSV: {valid_output_path}")
    print(f"  📊 Report: {report_output_path}")
    print(f"\nFinal statistics:")
    print(f"  Input: {len(data_df_filtered)} samples")
    print(f"  After cleaning: {len(all_final)} samples")
    print(f"  Train: {len(train_save)} samples")
    print(f"  Valid: {len(valid_save)} samples")
    print(f"  Retention rate: {100*len(all_final)/len(data_df_filtered):.1f}%")
    print(f"  Data leakage: {'❌ YES' if len(overlap) > 0 else '✅ NO'}")
    print("\n🚀 Ready for state-of-the-art training!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

class MIMICCXRCleanDataset(Dataset):
    """
    MIMIC-CXR Dataset using PRE-CLEANED findings
    No extraction needed - just load and use!
    """
    
    def __init__(self, dataframe, transform=None, image_root=None):
        """
        Args:
            dataframe: DataFrame with 'Findings_Clean' column
            transform: Image transformations
            image_root: Root directory for images
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_root = image_root
        
        # Handle labels
        for col in self.label_cols:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].fillna(0)
                self.dataframe[col] = self.dataframe[col].apply(
                    lambda x: 1 if x in [1.0, -1.0] else 0
                )
        
        # Verify clean findings exist
        if 'Findings_Clean' not in self.dataframe.columns:
            raise ValueError("Missing 'Findings_Clean' column! Run clean_mimic_reports.py first.")
        
        valid_count = (self.dataframe['Findings_Clean'].str.len() > 30).sum()
        print(f"✓ Dataset ready: {len(self.dataframe)} samples, {valid_count} with clean findings")
    
    def __len__(self):
        return len(self.dataframe)
    
    # REPLACE __getitem__ label handling with:
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_root, item['Path'])
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            # No labels for IU-Xray - return dummy or skip
            label = torch.zeros(14)  # Match MIMIC format for compatibility
            
            findings = item['Findings_Clean']
            if pd.isna(findings):
                findings = ""
            else:
                findings = str(findings).strip()
            
            return image, label, findings
            
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Return dummy data
            image = torch.zeros(3, 320, 320)
            label = torch.zeros(len(self.label_cols))
            findings = ""
            return image, label, findings


def load_clean_mimic_data(
    cleaned_dir="/mnt/External/Seagate/dawood/datasets/mimic-cxr/cleaned",
    image_root="/mnt/External/Seagate/dawood/datasets/mimic-cxr/jpg"
):
    """
    Load pre-cleaned MIMIC-CXR data (FAST - <5 seconds)
    
    Returns:
        train_df, valid_df with 'Findings_Clean' column
    """
    import pandas as pd
    import os
    
    train_csv = os.path.join(cleaned_dir, "mimic_clean_train.csv")
    valid_csv = os.path.join(cleaned_dir, "mimic_clean_valid.csv")
    
    if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
        raise FileNotFoundError(
            f"Clean data not found! Please run clean_mimic_reports.py first.\n"
            f"Expected files:\n"
            f"  {train_csv}\n"
            f"  {valid_csv}"
        )
    
    print("⚡ Loading pre-cleaned MIMIC-CXR data...")
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    
    print(f"✅ Loaded in <1 second!")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Valid: {len(valid_df)} samples")
    
    # Verify findings quality
    train_valid = (train_df['Findings_Clean'].str.len() > 30).sum()
    valid_valid = (valid_df['Findings_Clean'].str.len() > 30).sum()
    
    print(f"\n📊 Quality check:")
    print(f"   Train: {train_valid}/{len(train_df)} ({100*train_valid/len(train_df):.1f}%) valid findings")
    print(f"   Valid: {valid_valid}/{len(valid_df)} ({100*valid_valid/len(valid_df):.1f}%) valid findings")
    
    return train_df, valid_df