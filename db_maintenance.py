import sqlite3
import os
import logging
import json
import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from advanced_translation_system import AdvancedTranslationSystem, decode_unicode

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_database(db_path='translation_memory.db'):
    """Analyze the translation memory database and report statistics."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get total entries
        cursor.execute("SELECT COUNT(*) FROM translations")
        total_entries = cursor.fetchone()[0]
        print(f"Total translation entries: {total_entries}")
        
        # Count entries with each language
        cursor.execute("SELECT COUNT(*) FROM translations WHERE english IS NOT NULL AND english != ''")
        english_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM translations WHERE tamil IS NOT NULL AND tamil != ''")
        tamil_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM translations WHERE hindi IS NOT NULL AND hindi != ''")
        hindi_entries = cursor.fetchone()[0]
        
        print(f"English entries: {english_entries} ({english_entries/total_entries:.1%})")
        print(f"Tamil entries: {tamil_entries} ({tamil_entries/total_entries:.1%})")
        print(f"Hindi entries: {hindi_entries} ({hindi_entries/total_entries:.1%})")
        
        # ... rest of the function using print instead of logger.info ...
    except Exception as e:
        print(f"Error analyzing database: {e}")

 