import os
import json
import logging
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sarvam API Configuration
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "b61ffcf0-9e8f-498e-bb5d-4b7f8eb70132")
SARVAM_API_ENDPOINT = "https://api.sarvam.ai/translate"

# Utility function to decode Unicode escape sequences
def decode_unicode(text):
    """Decode Unicode escape sequences in text."""
    if not text or not isinstance(text, str):
        return text
    
    try:
        # Check if the text contains Unicode escape sequences
        if '\\u' in text:
            # Use a safer approach to handle potentially malformed Unicode
            result = ''
            i = 0
            while i < len(text):
                if text[i:i+2] == '\\u' and i + 6 <= len(text):
                    try:
                        # Extract 4 hex digits and convert to Unicode character
                        hex_val = int(text[i+2:i+6], 16)
                        result += chr(hex_val)
                        i += 6
                    except ValueError:
                        # If not a valid hex sequence, treat as regular characters
                        result += text[i]
                        i += 1
                else:
                    result += text[i]
                    i += 1
            return result
        return text
    except Exception as e:
        logger.error(f"Error decoding Unicode: {str(e)}")
        # Return original text on error instead of None
        return text

class SimpleTranslationSystem:
    """Simplified Translation System focused on database lookups and Sarvam AI API."""
    
    def __init__(self, db_path='translation_memory.db'):
        """Initialize the translation system.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        
        # Setup database
        self.setup_database()
        
        # Define supported languages
        self.supported_languages = {
            'en-IN': 'English',
            'hi-IN': 'Hindi',
            'ta-IN': 'Tamil'
        }
        
        # Load initial data from files
        self.load_translation_files()
    
    def setup_database(self):
        """Create SQLite database and tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            english TEXT,
            tamil TEXT,
            hindi TEXT,
            context TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database setup complete at {self.db_path}")
    
    def load_translation_files(self):
        """Load translations from .properties and .tmx files."""
        # Load from properties files
        properties_loaded = self.load_properties_files()
        
        # Load from TMX files
        tmx_loaded = self.load_tmx_files()
        
        logger.info(f"Loaded {properties_loaded} entries from properties files and {tmx_loaded} entries from TMX files")
    
    def load_properties_files(self):
        """Load translations from .properties files."""
        count = 0
        property_files = {
            'en-IN': list(Path('.').glob('*_en.properties')),
            'hi-IN': list(Path('.').glob('*_hi.properties')),
            'ta-IN': list(Path('.').glob('*_ta.properties'))
        }
        
        # Group files by their base name to align translations
        base_names = set()
        for files in property_files.values():
            for file_path in files:
                base_name = file_path.stem.split('_')[0]
                base_names.add(base_name)
        
        for base_name in base_names:
            translations = {}
            
            # Parse each language file for this base name
            for lang, files in property_files.items():
                lang_file = next((f for f in files if f.stem.startswith(f"{base_name}_")), None)
                if lang_file:
                    translations[lang] = self.parse_properties_file(lang_file)
            
            # Store aligned translations
            count += self.align_and_store_translations(translations)
        
        return count
    
    def parse_properties_file(self, filepath):
        """Parse a .properties file and return key-value pairs."""
        translations = {}
        
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                    
                # Split at the first '=' only
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key, value = parts
                    translations[key.strip()] = value
        
        return translations
    
    def load_tmx_files(self):
        """Load translations from .tmx files."""
        count = 0
        tmx_files = list(Path('.').glob('*.tmx'))
        
        for tmx_file in tmx_files:
            translations = self.parse_tmx_file(tmx_file)
            count += len(translations)
            
            # Store TMX translations
            for entry in translations:
                self.store_translation_triplet(**entry)
        
        return count
    
    def parse_tmx_file(self, filepath):
        """Parse a TMX file and extract translations."""
        translations = []
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            for tu in root.findall('.//tu'):
                entry = {'english': None, 'tamil': None, 'hindi': None, 'context': None}
                
                # Extract context if available
                if 'context' in tu.attrib:
                    entry['context'] = tu.attrib['context']
                
                # Process each translation unit variant
                for tuv in tu.findall('.//tuv'):
                    lang_code = tuv.get('{http://www.w3.org/XML/1998/namespace}lang', '')
                    seg = tuv.find('.//seg')
                    
                    if seg is not None and seg.text:
                        if lang_code.startswith('en'):
                            entry['english'] = seg.text
                        elif lang_code.startswith('ta'):
                            entry['tamil'] = seg.text
                        elif lang_code.startswith('hi'):
                            entry['hindi'] = seg.text
                
                # Add if we have at least source and one target
                if entry['english'] and (entry['tamil'] or entry['hindi']):
                    translations.append(entry)
            
            return translations
        except Exception as e:
            logger.error(f"Error parsing TMX file {filepath}: {str(e)}")
            return []
    
    def align_and_store_translations(self, translations):
        """Align translations by key across languages and store in database."""
        count = 0
        
        # Find common keys across all language files
        all_keys = set()
        for lang, trans_dict in translations.items():
            all_keys.update(trans_dict.keys())
        
        # For each key, store a translation triplet
        for key in all_keys:
            triplet = {
                'english': translations.get('en-IN', {}).get(key),
                'tamil': translations.get('ta-IN', {}).get(key),
                'hindi': translations.get('hi-IN', {}).get(key),
                'context': key  # Use the property key as context
            }
            
            # Store if we have at least source and one target language
            if triplet['english'] and (triplet['tamil'] or triplet['hindi']):
                self.store_translation_triplet(**triplet)
                count += 1
        
        return count
    
    def store_translation_triplet(self, english=None, tamil=None, hindi=None, context=None):
        """Store a translation triplet in the database."""
        try:
            # Clean and normalize inputs
            if english:
                english = english.strip()
            if tamil:
                tamil = tamil.strip()
            if hindi:
                hindi = hindi.strip()
            
            # Skip empty inputs
            if not any([english, tamil, hindi]):
                logger.warning("Attempted to store empty translation triplet")
                return False
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if this exact English sentence already exists
            id = None
            if english:
                cursor.execute("SELECT id FROM translations WHERE english = ?", (english,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing entry
                    id = existing[0]
                    cursor.execute(
                        """UPDATE translations SET 
                        tamil = CASE WHEN ? IS NOT NULL THEN ? ELSE tamil END, 
                        hindi = CASE WHEN ? IS NOT NULL THEN ? ELSE hindi END, 
                        context = CASE WHEN ? IS NOT NULL THEN ? ELSE context END 
                        WHERE id = ?""",
                        (tamil, tamil, hindi, hindi, context, context, id)
                    )
                    logger.info(f"Updated translation triplet (ID: {id})")
                else:
                    # Insert new entry
                    cursor.execute(
                        "INSERT INTO translations (english, tamil, hindi, context) VALUES (?, ?, ?, ?)",
                        (english, tamil, hindi, context)
                    )
                    id = cursor.lastrowid
                    logger.info(f"Stored new translation triplet (ID: {id})")
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error storing translation: {str(e)}")
            try:
                conn.rollback()
                conn.close()
            except:
                pass
            return False
    
    def search_exact_match(self, text, source_lang, target_lang):
        """Search for an exact match in the translation memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        source_field = self.get_field_name(source_lang)
        target_field = self.get_field_name(target_lang)
        
        # First try with exact match
        query = f"SELECT {target_field} FROM translations WHERE {source_field} = ? COLLATE NOCASE"
        cursor.execute(query, (text,))
        row = cursor.fetchone()
        
        if row and row[0]:
            conn.close()
            return decode_unicode(row[0]), 1.0
        
        conn.close()
        return None, 0.0
    
    def get_field_name(self, lang_code):
        """Get the field name in the database based on language code."""
        if lang_code.startswith('en'):
            return 'english'
        if lang_code.startswith('ta'):
            return 'tamil'
        if lang_code.startswith('hi'):
            return 'hindi'
        return 'english'  # Default fallback
    
    def sarvam_translate(self, text, source_lang, target_lang):
        """Translate text using Sarvam AI API."""
        try:
            headers = {
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "output_script": None  # No transliteration, native script
            }
            
            response = requests.post(
                SARVAM_API_ENDPOINT,
                headers=headers,
                data=json.dumps(payload)
            )
            
            response.raise_for_status()
            result = response.json()
            
            translated_text = result.get("translated_text", text)
            logger.info(f"Sarvam API translation: {text} -> {translated_text}")
            
            return translated_text
        except Exception as e:
            logger.error(f"Sarvam API error: {str(e)}")
            return text
    
    def translate(self, text, source_lang, target_lang):
        """Translate text from source language to target language."""
        if not text:
            return ""
        
        if source_lang == target_lang:
            return text
        
        # Check if language is supported
        if source_lang not in self.supported_languages or target_lang not in self.supported_languages:
            logger.warning(f"Unsupported language pair: {source_lang} -> {target_lang}")
            return text
        
        # Track full phrase for logging
        original_text = text.strip()
        
        # Step 1: Try exact match for the complete phrase from database (highest priority)
        exact_match, exact_score = self.search_exact_match(original_text, source_lang, target_lang)
        
        if exact_match:
            logger.info(f"Exact match translation: '{original_text}' -> '{exact_match}'")
            return exact_match
        
        # Step 2: Fallback to Sarvam AI API
        try:
            api_translation = self.sarvam_translate(original_text, source_lang, target_lang)
            
            # Store the API translation for future use
            triplet = {
                'english': original_text if source_lang == 'en-IN' else api_translation if target_lang == 'en-IN' else None,
                'tamil': api_translation if target_lang == 'ta-IN' else original_text if source_lang == 'ta-IN' else None,
                'hindi': api_translation if target_lang == 'hi-IN' else original_text if source_lang == 'hi-IN' else None,
                'context': f"Sarvam API translation"
            }
            
            self.store_translation_triplet(**triplet)
            logger.info(f"API translation stored: '{original_text}' -> '{api_translation}'")
            
            return api_translation
        except Exception as e:
            logger.error(f"Error in API translation: {str(e)}")
            return original_text  # Return original text if all else fails
    
    def get_all_triplets(self):
        """Get all translation triplets from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, english, tamil, hindi, context, timestamp FROM translations")
        rows = cursor.fetchall()
        conn.close()
        
        triplets = []
        for row in rows:
            triplets.append({
                'id': row[0],
                'english': row[1],
                'tamil': decode_unicode(row[2]),
                'hindi': decode_unicode(row[3]),
                'context': row[4],
                'timestamp': row[5]
            })
        
        return triplets

def main():
    """Test the translation system."""
    print("🚀 Starting Simple Translation System Test")
    print("=" * 50)
    
    # Initialize the translation system
    translator = SimpleTranslationSystem()
    
    # Display database statistics
    triplets = translator.get_all_triplets()
    print(f"📊 Database contains {len(triplets)} translation entries")
    
    # Count by language
    en_count = sum(1 for t in triplets if t['english'])
    ta_count = sum(1 for t in triplets if t['tamil'])
    hi_count = sum(1 for t in triplets if t['hindi'])
    
    print(f"   - English entries: {en_count}")
    print(f"   - Tamil entries: {ta_count}")
    print(f"   - Hindi entries: {hi_count}")
    print()
    
    # Test translations
    print("🔄 Testing translations:")
    print("-" * 30)
    
    test_cases = [
        # Database lookups (should be fast)
        ("app.title", "en-IN", "hi-IN"),
        ("navigation.inbox", "en-IN", "ta-IN"),
        ("compose.send", "en-IN", "hi-IN"),
        
        # Simple words/phrases (may use API)
        ("Hello", "en-IN", "hi-IN"),
        ("Thank you", "en-IN", "ta-IN"),
        ("Good morning", "en-IN", "hi-IN"),
        
        # Reverse translations
        ("मेल", "hi-IN", "en-IN"),
        ("ஆவணம்", "ta-IN", "en-IN"),
        
        # New phrases (will use API)
        ("How are you?", "en-IN", "ta-IN"),
        ("I am fine", "en-IN", "hi-IN"),
    ]
    
    for i, (text, src_lang, tgt_lang) in enumerate(test_cases, 1):
        print(f"{i:2d}. {src_lang} → {tgt_lang}")
        print(f"    Input:  '{text}'")
        
        try:
            translation = translator.translate(text, src_lang, tgt_lang)
            print(f"    Output: '{translation}'")
            
            # Check if it's different from input (successful translation)
            if translation != text:
                print(f"    Status: ✅ Translation successful")
            else:
                print(f"    Status: ⚠️  No translation (returned original)")
                
        except Exception as e:
            print(f"    Status: ❌ Error - {str(e)}")
        
        print()
    
    # Final statistics
    updated_triplets = translator.get_all_triplets()
    new_entries = len(updated_triplets) - len(triplets)
    
    print("=" * 50)
    print(f"📈 Test completed!")
    print(f"   - New entries added: {new_entries}")
    print(f"   - Total entries now: {len(updated_triplets)}")
    print("=" * 50)

if __name__ == "__main__":
    main() 