import sqlite3
import pandas as pd
import re
import streamlit as st
from pathlib import Path
import time
from difflib import SequenceMatcher
import logging
import requests
import json
from functools import lru_cache

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sarvam API Configuration
SARVAM_API_KEY = "b61ffcf0-9e8f-498e-bb5d-4b7f8eb70132"
SARVAM_API_ENDPOINT = "https://api.sarvam.ai/translate"

class TranslationMemory:
    def __init__(self, db_path='translation_memory.db'):
        self.db_path = db_path
        try:
            # Try to delete existing database if it exists
            if Path(db_path).exists():
                try:
                    Path(db_path).unlink()
                except PermissionError:
                    logger.warning("Could not delete existing database - may be in use. Will try to use existing database.")
            self.setup_database()
            self.load_initial_data()
            
            # Load translations from properties and TMX files
            self.load_properties_files()
            self.load_tmx_files()
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def setup_database(self):
        """Create SQLite database and tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='translations'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            cursor.execute('''
            CREATE TABLE translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_text TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_text TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                is_sentence INTEGER DEFAULT 0,
                context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            logger.info("Created new translations table with schema")
        else:
            # Check if is_sentence column exists
            cursor.execute("PRAGMA table_info(translations)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'is_sentence' not in columns:
                cursor.execute('ALTER TABLE translations ADD COLUMN is_sentence INTEGER DEFAULT 0')
                logger.info("Added is_sentence column to existing table")
            if 'context' not in columns:
                cursor.execute('ALTER TABLE translations ADD COLUMN context TEXT')
                logger.info("Added context column to existing table")
                
        conn.commit()
        conn.close()

    def parse_properties_file(self, filepath):
        """Parse Java properties files and extract translations"""
        translations = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                current_lang = self._detect_language_from_filename(filepath)
                target_lang = 'en-IN'  # Default target language
                
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            # Store both the key and translated value
                            translations.append({
                                'source_text': value.strip(),
                                'source_lang': current_lang,
                                'target_text': key.strip(),
                                'target_lang': target_lang,
                                'is_sentence': len(value.split()) > 1  # Consider as sentence if more than one word
                            })
            return translations
        except Exception as e:
            logger.error(f"Error parsing properties file {filepath}: {str(e)}")
            return []

    def _detect_language_from_filename(self, filepath):
        """Detect language code from filename"""
        filename = Path(filepath).stem
        if '_hi' in filename:
            return 'hi-IN'
        elif '_ta' in filename:
            return 'ta-IN'
        return 'en-IN'  # Default to English

    def load_properties_files(self):
        """Load all properties files from the current directory"""
        try:
            property_files = list(Path('.').glob('*.properties'))
            total_translations = 0
            
            for prop_file in property_files:
                translations = self.parse_properties_file(prop_file)
                for trans in translations:
                    self.store_translation(
                        trans['source_text'],
                        trans['source_lang'],
                        trans['target_text'],
                        trans['target_lang'],
                        trans['is_sentence']
                    )
                total_translations += len(translations)
                
            logger.info(f"Loaded {total_translations} translations from {len(property_files)} properties files")
        except Exception as e:
            logger.error(f"Error loading properties files: {str(e)}")

    def load_tmx_files(self):
        """Load translations from TMX files"""
        try:
            tmx_files = list(Path('.').glob('*.tmx'))
            total_translations = 0
            
            for tmx_file in tmx_files:
                translations = self.parse_tmx_file(tmx_file)
                for trans in translations:
                    self.store_translation(
                        trans['source_text'],
                        trans['source_lang'],
                        trans['target_text'],
                        trans['target_lang'],
                        trans['is_sentence']
                    )
                total_translations += len(translations)
                
            logger.info(f"Loaded {total_translations} translations from {len(tmx_files)} TMX files")
        except Exception as e:
            logger.error(f"Error loading TMX files: {str(e)}")

    def parse_tmx_file(self, filepath):
        """Parse TMX file and extract translations"""
        translations = []
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            for tu in root.findall('.//tu'):
                source_text = None
                target_text = None
                source_lang = None
                target_lang = None
                
                for tuv in tu.findall('tuv'):
                    lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang', '')
                    seg = tuv.find('seg')
                    if seg is not None and seg.text:
                        if source_text is None:
                            source_text = seg.text
                            source_lang = lang
                        else:
                            target_text = seg.text
                            target_lang = lang
                
                if source_text and target_text and source_lang and target_lang:
                    translations.append({
                        'source_text': source_text,
                        'source_lang': self._normalize_language_code(source_lang),
                        'target_text': target_text,
                        'target_lang': self._normalize_language_code(target_lang),
                        'is_sentence': True  # TMX entries are usually sentences
                    })
            
            return translations
        except Exception as e:
            logger.error(f"Error parsing TMX file {filepath}: {str(e)}")
            return []

    def _normalize_language_code(self, lang_code):
        """Normalize language codes to our format"""
        lang_map = {
            'en': 'en-IN',
            'hi': 'hi-IN',
            'ta': 'ta-IN',
            'eng': 'en-IN',
            'hin': 'hi-IN',
            'tam': 'ta-IN'
        }
        # Remove any region codes and lowercase
        base_code = lang_code.lower().split('-')[0].split('_')[0]
        return lang_map.get(base_code, 'en-IN')

    def load_initial_data(self):
        """Load initial translations from properties files"""
        try:
            # Load common translations
            initial_translations = [
                # Tamil to English
                ("நான்", "ta-IN", "I", "en-IN", False),
                ("என்", "ta-IN", "My", "en-IN", False),
                ("நீங்கள்", "ta-IN", "You", "en-IN", False),
                ("அவன்", "ta-IN", "He", "en-IN", False),
                ("அவள்", "ta-IN", "She", "en-IN", False),
                ("அது", "ta-IN", "It", "en-IN", False),
                ("ஆம்", "ta-IN", "Yes", "en-IN", False),
                ("இல்லை", "ta-IN", "No", "en-IN", False),
                ("நன்றி", "ta-IN", "Thank you", "en-IN", False),
                ("வணக்கம்", "ta-IN", "Hello", "en-IN", False),
                
                # Hindi to English
                ("मैं", "hi-IN", "I", "en-IN", False),
                ("मेरा", "hi-IN", "My", "en-IN", False),
                ("आप", "hi-IN", "You", "en-IN", False),
                ("वह", "hi-IN", "He/She", "en-IN", False),
                ("हाँ", "hi-IN", "Yes", "en-IN", False),
                ("नहीं", "hi-IN", "No", "en-IN", False),
                ("धन्यवाद", "hi-IN", "Thank you", "en-IN", False),
                ("नमस्ते", "hi-IN", "Hello", "en-IN", False),
            ]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for source_text, source_lang, target_text, target_lang, is_sentence in initial_translations:
                cursor.execute('''
                INSERT OR IGNORE INTO translations 
                (source_text, source_lang, target_text, target_lang, is_sentence)
                VALUES (?, ?, ?, ?, ?)
                ''', (source_text, source_lang, target_text, target_lang, is_sentence))
            
            conn.commit()
            conn.close()
            logger.info("Loaded initial translations into database")
            
        except Exception as e:
            logger.error(f"Error loading initial data: {str(e)}")

    def get_translation(self, text, source_lang, target_lang, min_ratio=0.8):
        """Get translation from memory with context-aware fuzzy matching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First try exact match for full sentences
        cursor.execute('''
        SELECT target_text FROM translations 
        WHERE source_text = ? AND source_lang = ? AND target_lang = ? AND is_sentence = 1
        ''', (text, source_lang, target_lang))
        
        result = cursor.fetchone()
        if result:
            logger.debug(f"Found exact sentence match: {result[0]}")
            conn.close()
            return result[0]

        # Try fuzzy matching for sentences
        cursor.execute('''
        SELECT source_text, target_text FROM translations 
        WHERE source_lang = ? AND target_lang = ? AND is_sentence = 1
        ''', (source_lang, target_lang))
        
        matches = cursor.fetchall()
        best_ratio = 0
        best_match = None
        
        for source, target in matches:
            ratio = SequenceMatcher(None, text.lower(), source.lower()).ratio()
            if ratio > min_ratio and ratio > best_ratio:
                best_ratio = ratio
                best_match = target
                logger.debug(f"Found fuzzy sentence match: {source} -> {target} (ratio: {ratio})")

        if best_match:
            conn.close()
            return best_match

        # If no sentence match found, try word lookup as last resort
        words = text.split()
        if len(words) == 1:
            cursor.execute('''
            SELECT target_text FROM translations 
            WHERE source_text = ? AND source_lang = ? AND target_lang = ? AND is_sentence = 0
            ''', (text, source_lang, target_lang))
            
            result = cursor.fetchone()
            if result:
                logger.debug(f"Found word match: {result[0]}")
                conn.close()
                return result[0]

        conn.close()
        return None

    def get_all_translations(self):
        """Get all stored translations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            SELECT source_text, source_lang, target_text, target_lang, is_sentence, timestamp 
            FROM translations 
            ORDER BY timestamp DESC
            ''')
            translations = cursor.fetchall()
            conn.close()
            return translations
        except Exception as e:
            logger.error(f"Error fetching translations: {str(e)}")
            return []

    def store_translation(self, source_text, source_lang, target_text, target_lang, is_sentence=False):
        """Store translation in memory with sentence flag and better error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if translation already exists
            cursor.execute('''
            SELECT id FROM translations 
            WHERE source_text = ? AND source_lang = ? AND target_lang = ?
            ''', (source_text, source_lang, target_lang))
            
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute('''
                UPDATE translations 
                SET target_text = ?, is_sentence = ?, timestamp = CURRENT_TIMESTAMP
                WHERE id = ?
                ''', (target_text, is_sentence, existing[0]))
                logger.info(f"Updated existing translation: {source_text} -> {target_text}")
            else:
                cursor.execute('''
                INSERT INTO translations (source_text, source_lang, target_text, target_lang, is_sentence)
                VALUES (?, ?, ?, ?, ?)
                ''', (source_text, source_lang, target_text, target_lang, is_sentence))
                logger.info(f"Stored new translation: {source_text} -> {target_text}")
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error storing translation: {str(e)}")
            return False

class TranslationAgent:
    def __init__(self):
        self.tm = TranslationMemory()
        self.supported_languages = {
            'en-IN': 'English',
            'hi-IN': 'Hindi',
            'ta-IN': 'Tamil'
        }

    def translate_text(self, text, source_lang, target_lang):
        """Translate text with improved linguistic handling"""
        if not text.strip():
            return ""

        logger.debug(f"\nTranslating: {text}")
        logger.debug(f"From {source_lang} to {target_lang}")

        # First try to get a full sentence translation from memory
        translation = self.tm.get_translation(text, source_lang, target_lang)
        if translation:
            logger.debug(f"Using translation from memory: {translation}")
            return translation

        # If not found as a full sentence, try to translate word by word from our database
        words = text.split()
        if len(words) > 1:
            translated_words = []
            all_words_found = True
            
            for word in words:
                word_trans = self.tm.get_translation(word, source_lang, target_lang)
                if word_trans:
                    translated_words.append(word_trans)
                else:
                    all_words_found = False
                    break
            
            if all_words_found:
                translation = ' '.join(translated_words)
                logger.debug(f"Using word-by-word translation from memory: {translation}")
                # Store this sentence translation for future use
                self.tm.store_translation(text, source_lang, translation, target_lang, is_sentence=True)
                return translation

        # If not in memory, use Sarvam API for full sentence
        try:
            logger.debug("No translation found in memory, using Sarvam API")
            headers = {
                'api-subscription-key': SARVAM_API_KEY,
                'Content-Type': 'application/json'
            }
            
            data = {
                'input': text,
                'source_language_code': source_lang,
                'target_language_code': target_lang,
                'mode': 'formal'
            }
            
            logger.debug(f"Sending request to Sarvam API: {data}")
            response = requests.post(SARVAM_API_ENDPOINT, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result['translated_text']
            
            # Store the full sentence translation
            self.tm.store_translation(text, source_lang, translated_text, target_lang, is_sentence=True)
            
            # Also store individual word translations if they don't exist
            if len(words) > 1:
                source_words = text.split()
                target_words = translated_text.split()
                if len(source_words) == len(target_words):
                    for src_word, tgt_word in zip(source_words, target_words):
                        if not self.tm.get_translation(src_word, source_lang, target_lang):
                            self.tm.store_translation(src_word, source_lang, tgt_word, target_lang, is_sentence=False)
            
            logger.debug(f"API translation: {translated_text}")
            return translated_text

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response content: {e.response.content}")
            return f"Translation error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Multi-Language Translator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Multi-Language Translator")
    st.sidebar.title("Settings")

    translator = TranslationAgent()
    
    # Add tabs for different views
    tab1, tab2 = st.tabs(["Translate", "View Translations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "From Language",
                options=list(translator.supported_languages.keys()),
                format_func=lambda x: translator.supported_languages[x]
            )
        
        with col2:
            target_options = [lang for lang in translator.supported_languages.keys() if lang != source_lang]
            target_lang = st.selectbox(
                "To Language",
                options=target_options,
                format_func=lambda x: translator.supported_languages[x]
            )

        text_to_translate = st.text_area("Enter text to translate:", height=100)
        
        if st.button("Translate"):
            if text_to_translate:
                with st.spinner("Translating..."):
                    # Auto-detect Tamil script and override source language if needed
                    if any('\u0B80' <= c <= '\u0BFF' for c in text_to_translate):
                        source_lang = 'ta-IN'
                        st.info("Detected Tamil text - automatically setting source language to Tamil")
                    elif any('\u0900' <= c <= '\u097F' for c in text_to_translate):
                        source_lang = 'hi-IN'
                        st.info("Detected Hindi text - automatically setting source language to Hindi")
                    
                    translation = translator.translate_text(text_to_translate, source_lang, target_lang)
                    st.success("Translation:")
                    st.write(translation)
            else:
                st.warning("Please enter some text to translate.")
    
    with tab2:
        st.subheader("Stored Translations")
        translations = translator.tm.get_all_translations()
        
        if translations:
            df = pd.DataFrame(translations, 
                            columns=['Source Text', 'Source Language', 'Target Text', 
                                   'Target Language', 'Is Sentence', 'Timestamp'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No translations stored yet.")
    
    # Add about section in sidebar
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This translator uses:
        1. Context-aware sentence translation
        2. Linguistic pattern matching
        3. Sarvam AI Translation API
        4. Local translation memory
        """
    )
    
    # Add statistics in sidebar
    st.sidebar.markdown("### Statistics")
    try:
        conn = sqlite3.connect('translation_memory.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM translations')
        total = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM translations WHERE is_sentence = 1')
        sentences = c.fetchone()[0]
        st.sidebar.metric("Total Translations", total)
        st.sidebar.metric("Stored Sentences", sentences)
        conn.close()
    except Exception as e:
        st.sidebar.error("Could not load statistics")

if __name__ == "__main__":
    main()