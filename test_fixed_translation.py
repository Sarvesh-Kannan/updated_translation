#!/usr/bin/env python3
"""
Test script for fixed translation system with improved matching algorithms
"""

import os
import json
import logging
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sarvam API Configuration
SARVAM_API_KEY = "b61ffcf0-9e8f-498e-bb5d-4b7f8eb70132"
SARVAM_API_ENDPOINT = "https://api.sarvam.ai/translate"

def decode_unicode(text):
    """Decode Unicode escape sequences in text."""
    if not text or not isinstance(text, str):
        return text
    
    try:
        if '\\u' in text:
            result = ''
            i = 0
            while i < len(text):
                if text[i:i+2] == '\\u' and i + 6 <= len(text):
                    try:
                        hex_val = int(text[i+2:i+6], 16)
                        result += chr(hex_val)
                        i += 6
                    except ValueError:
                        result += text[i]
                        i += 1
                else:
                    result += text[i]
                    i += 1
            return result
        return text
    except Exception as e:
        logger.error(f"Error decoding Unicode: {str(e)}")
        return text

class FixedTranslationSystem:
    """Fixed Translation System with corrected index mapping."""
    
    def __init__(self, db_path='translation_memory.db'):
        self.db_path = db_path
        self.supported_languages = {
            'en-IN': 'English',
            'hi-IN': 'Hindi', 
            'ta-IN': 'Tamil'
        }
    
    def get_field_name(self, lang_code):
        """Get the field name in the database based on language code."""
        if lang_code.startswith('en'):
            return 'english'
        if lang_code.startswith('ta'):
            return 'tamil'
        if lang_code.startswith('hi'):
            return 'hindi'
        return 'english'
    
    def search_exact_match(self, text, source_lang, target_lang):
        """Search for an exact match in the translation memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            source_field = self.get_field_name(source_lang)
            target_field = self.get_field_name(target_lang)
            
            # Try exact match (case insensitive)
            query = f"SELECT {target_field} FROM translations WHERE {source_field} = ? COLLATE NOCASE"
            cursor.execute(query, (text,))
            row = cursor.fetchone()
            
            if row and row[0]:
                conn.close()
                return decode_unicode(row[0]), 1.0
            
            conn.close()
            return None, 0.0
        except Exception as e:
            logger.error(f"Error in exact match search: {str(e)}")
            return None, 0.0
    
    def search_fuzzy_match(self, text, source_lang, target_lang, min_similarity=0.8):
        """Search for fuzzy matches using improved string similarity."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            source_field = self.get_field_name(source_lang)
            target_field = self.get_field_name(target_lang)
            
            # Get all candidates
            query = f"SELECT {source_field}, {target_field} FROM translations WHERE {source_field} IS NOT NULL AND {target_field} IS NOT NULL"
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            best_match = None
            best_score = 0.0
            
            text_lower = text.lower().strip()
            
            for source_text, target_text in rows:
                if not source_text or not target_text:
                    continue
                    
                source_lower = source_text.lower().strip()
                
                # Calculate similarity using SequenceMatcher
                similarity = SequenceMatcher(None, text_lower, source_lower).ratio()
                
                # Check for substring matches
                if text_lower in source_lower or source_lower in text_lower:
                    substring_score = min(len(text_lower), len(source_lower)) / max(len(text_lower), len(source_lower))
                    similarity = max(similarity, substring_score * 0.9)
                
                # Check for word-level matches
                text_words = set(text_lower.split())
                source_words = set(source_lower.split())
                if text_words and source_words:
                    word_overlap = len(text_words.intersection(source_words)) / len(text_words.union(source_words))
                    similarity = max(similarity, word_overlap * 0.8)
                
                if similarity >= min_similarity and similarity > best_score:
                    best_score = similarity
                    best_match = decode_unicode(target_text)
                    logger.info(f"Fuzzy match found ({similarity:.3f}): '{text}' -> '{source_text}' -> '{target_text}'")
            
            if best_match:
                return best_match, best_score
            
            return None, 0.0
        except Exception as e:
            logger.error(f"Error in fuzzy match search: {str(e)}")
            return None, 0.0
    
    def search_partial_match(self, text, source_lang, target_lang):
        """Search for partial matches using SQL LIKE."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            source_field = self.get_field_name(source_lang)
            target_field = self.get_field_name(target_lang)
            
            # Search for partial matches
            query = f"SELECT {source_field}, {target_field} FROM translations WHERE {source_field} LIKE ? AND {target_field} IS NOT NULL"
            cursor.execute(query, (f"%{text}%",))
            rows = cursor.fetchall()
            
            if rows:
                # Return the first match (could be improved with scoring)
                source_text, target_text = rows[0]
                similarity = len(text) / len(source_text) if source_text else 0
                conn.close()
                logger.info(f"Partial match found ({similarity:.3f}): '{text}' in '{source_text}' -> '{target_text}'")
                return decode_unicode(target_text), similarity
            
            conn.close()
            return None, 0.0
        except Exception as e:
            logger.error(f"Error in partial match search: {str(e)}")
            return None, 0.0
    
    def translate_word_by_word(self, text, source_lang, target_lang):
        """Translate text word by word and combine results."""
        words = text.split()
        translations = []
        
        for word in words:
            # Clean the word (remove punctuation)
            clean_word = word.strip('.,!?;:"()[]{}')
            
            # Try to translate each word
            exact_match, _ = self.search_exact_match(clean_word, source_lang, target_lang)
            if exact_match:
                translations.append(exact_match)
            else:
                # Try fuzzy match for single words
                fuzzy_match, fuzzy_score = self.search_fuzzy_match(clean_word, source_lang, target_lang, min_similarity=0.9)
                if fuzzy_match and fuzzy_score >= 0.95:
                    translations.append(fuzzy_match)
                else:
                    translations.append(word)  # Keep original if no translation found
        
        return " ".join(translations)
    
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
                "output_script": None
            }
            
            response = requests.post(
                SARVAM_API_ENDPOINT,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
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
        """Translate text with improved accuracy using multiple strategies."""
        if not text or not text.strip():
            return ""
        
        if source_lang == target_lang:
            return text
        
        if source_lang not in self.supported_languages or target_lang not in self.supported_languages:
            logger.warning(f"Unsupported language pair: {source_lang} -> {target_lang}")
            return text
        
        text = text.strip()
        logger.info(f"Translating: '{text}' from {source_lang} to {target_lang}")
        
        # Strategy 1: Exact match (highest priority)
        exact_match, exact_score = self.search_exact_match(text, source_lang, target_lang)
        if exact_match:
            logger.info(f"✅ Exact match found: '{text}' -> '{exact_match}'")
            return exact_match
        
        # Strategy 2: High-confidence fuzzy match
        fuzzy_match, fuzzy_score = self.search_fuzzy_match(text, source_lang, target_lang, min_similarity=0.85)
        if fuzzy_match and fuzzy_score >= 0.9:
            logger.info(f"✅ High-confidence fuzzy match ({fuzzy_score:.3f}): '{text}' -> '{fuzzy_match}'")
            return fuzzy_match
        
        # Strategy 3: Partial match for shorter texts
        if len(text.split()) <= 3:
            partial_match, partial_score = self.search_partial_match(text, source_lang, target_lang)
            if partial_match and partial_score >= 0.7:
                logger.info(f"✅ Partial match ({partial_score:.3f}): '{text}' -> '{partial_match}'")
                return partial_match
        
        # Strategy 4: Word-by-word translation for multi-word phrases
        if len(text.split()) > 1:
            word_translation = self.translate_word_by_word(text, source_lang, target_lang)
            if word_translation != text:  # If any words were translated
                logger.info(f"✅ Word-by-word translation: '{text}' -> '{word_translation}'")
                return word_translation
        
        # Strategy 5: Lower-confidence fuzzy match
        if fuzzy_match and fuzzy_score >= 0.7:
            logger.info(f"⚠️ Lower-confidence fuzzy match ({fuzzy_score:.3f}): '{text}' -> '{fuzzy_match}'")
            return fuzzy_match
        
        # Strategy 6: API fallback
        try:
            api_translation = self.sarvam_translate(text, source_lang, target_lang)
            if api_translation and api_translation != text:
                logger.info(f"✅ API translation: '{text}' -> '{api_translation}'")
                return api_translation
        except Exception as e:
            logger.error(f"API translation failed: {str(e)}")
        
        # Return original text if all strategies fail
        logger.warning(f"❌ No translation found for: '{text}'")
        return text
    
    def get_all_triplets(self):
        """Get all translation triplets from the database."""
        try:
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
                    'tamil': decode_unicode(row[2]) if row[2] else None,
                    'hindi': decode_unicode(row[3]) if row[3] else None,
                    'context': row[4],
                    'timestamp': row[5]
                })
            
            return triplets
        except Exception as e:
            logger.error(f"Error getting triplets: {str(e)}")
            return []
    
    def get_statistics(self):
        """Get system statistics."""
        triplets = self.get_all_triplets()
        total_entries = len(triplets)
        
        en_count = sum(1 for t in triplets if t.get('english'))
        hi_count = sum(1 for t in triplets if t.get('hindi'))
        ta_count = sum(1 for t in triplets if t.get('tamil'))
        
        return {
            'total_entries': total_entries,
            'english_entries': en_count,
            'hindi_entries': hi_count,
            'tamil_entries': ta_count
        }

def main():
    """Test the fixed translation system."""
    print("🔧 Testing Fixed Translation System")
    print("=" * 50)
    
    # Initialize the system
    translator = FixedTranslationSystem()
    
    # Get system statistics
    stats = translator.get_statistics()
    print(f"📊 System Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  English entries: {stats['english_entries']}")
    print(f"  Hindi entries: {stats['hindi_entries']}")
    print(f"  Tamil entries: {stats['tamil_entries']}")
    print()
    
    # Test problematic cases that were returning wrong translations
    test_cases = [
        ("Hello", "en-IN", "hi-IN"),
        ("Hello", "en-IN", "ta-IN"),
        ("Inbox", "en-IN", "ta-IN"),
        ("Important", "en-IN", "hi-IN"),
        ("Document", "en-IN", "ta-IN"),
        ("Before Zoho, we were chained to the desk", "en-IN", "ta-IN"),  # This was returning "வரைவுகள்"
        ("High priority attachments", "en-IN", "ta-IN"),
        ("The document needs to be signed", "en-IN", "ta-IN"),
        ("Save", "en-IN", "ta-IN"),
        ("Cancel", "en-IN", "ta-IN"),
        ("Submit", "en-IN", "hi-IN"),
        ("Search", "en-IN", "ta-IN"),
    ]
    
    print("🧪 Testing Translations with Fixed Algorithm:")
    print("-" * 50)
    
    for text, src, tgt in test_cases:
        print(f"\n🔍 Testing: '{text}' ({src} -> {tgt})")
        translation = translator.translate(text, src, tgt)
        print(f"   Result: '{translation}'")
        
        # Validate that we're not getting obviously wrong results
        if translation == "வரைவுகள்" and text != "Drafts":
            print("   ⚠️  WARNING: Possibly incorrect translation detected!")
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!")

if __name__ == "__main__":
    main()
