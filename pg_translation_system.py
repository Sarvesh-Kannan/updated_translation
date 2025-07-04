import json
import os
import requests
import psycopg2
from pathlib import Path
import logging
import time
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

# PostgreSQL Configuration
PG_HOST = "localhost"
PG_DATABASE = "translation_memory"
PG_USER = "postgres"
PG_PASSWORD = "1234"
PG_PORT = "5432"

class PostgresTranslationSystem:
    """
    A minimalist translation system with Translation Memory in PostgreSQL,
    in-context translation, and fallback to API translation.
    """
    
    def __init__(self):
        """Initialize the translation system with a PostgreSQL database for TM."""
        self.setup_database()
        
        self.supported_languages = {
            'en-IN': 'English',
            'hi-IN': 'Hindi',
            'ta-IN': 'Tamil'
        }
        
        # Min similarity ratio for fuzzy matching
        self.min_similarity = 0.85
    
    def get_connection(self):
        """Get a connection to the PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                host=PG_HOST,
                database=PG_DATABASE,
                user=PG_USER,
                password=PG_PASSWORD,
                port=PG_PORT
            )
            return conn
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            
            # If database doesn't exist, create it
            if "database" in str(e) and "does not exist" in str(e):
                try:
                    # Connect to default postgres database to create our database
                    conn = psycopg2.connect(
                        host=PG_HOST,
                        database="postgres",
                        user=PG_USER,
                        password=PG_PASSWORD,
                        port=PG_PORT
                    )
                    conn.autocommit = True
                    cursor = conn.cursor()
                    cursor.execute(f"CREATE DATABASE {PG_DATABASE}")
                    cursor.close()
                    conn.close()
                    
                    # Now connect to the new database
                    return psycopg2.connect(
                        host=PG_HOST,
                        database=PG_DATABASE,
                        user=PG_USER,
                        password=PG_PASSWORD,
                        port=PG_PORT
                    )
                except Exception as create_e:
                    logger.error(f"Error creating database: {str(create_e)}")
                    raise
            raise
    
    def setup_database(self):
        """Set up the PostgreSQL database for the translation memory."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create translation_triplets table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS translation_triplets (
                id SERIAL PRIMARY KEY,
                english TEXT,
                tamil TEXT,
                hindi TEXT,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create indices for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_english ON translation_triplets(english)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tamil ON translation_triplets(tamil)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hindi ON translation_triplets(hindi)')
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("PostgreSQL database setup complete")
        except Exception as e:
            logger.error(f"Error setting up database: {str(e)}")
            raise
    
    def _language_code_to_field(self, lang_code):
        """Convert language code to database field name."""
        mapping = {
            'en-IN': 'english',
            'ta-IN': 'tamil',
            'hi-IN': 'hindi'
        }
        return mapping.get(lang_code, 'english')
    
    def tm_lookup(self, text, source_lang, target_lang):
        """
        Look up a translation in the Translation Memory.
        
        Args:
            text (str): The text to translate
            source_lang (str): Source language code (e.g., 'en-IN')
            target_lang (str): Target language code (e.g., 'ta-IN')
            
        Returns:
            str or None: The translation if found, None otherwise
        """
        if not text.strip():
            return ""
        
        source_field = self._language_code_to_field(source_lang)
        target_field = self._language_code_to_field(target_lang)
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Try exact match first
            query = f"SELECT {target_field} FROM translation_triplets WHERE {source_field} = %s"
            cursor.execute(query, (text,))
            result = cursor.fetchone()
            
            if result and result[0]:
                logger.info(f"Exact match found in TM: {text} -> {result[0]}")
                cursor.close()
                conn.close()
                return result[0]
            
            # Try fuzzy matching if exact match fails
            query = f"SELECT {source_field}, {target_field} FROM translation_triplets WHERE {target_field} IS NOT NULL"
            cursor.execute(query)
            candidates = cursor.fetchall()
            cursor.close()
            conn.close()
            
            best_match = None
            best_ratio = 0
            
            for source, target in candidates:
                if source and target:  # Ensure neither is None
                    ratio = SequenceMatcher(None, text.lower(), source.lower()).ratio()
                    if ratio > self.min_similarity and ratio > best_ratio:
                        best_ratio = ratio
                        best_match = target
                        logger.debug(f"Fuzzy match ({ratio:.2f}): {source} -> {target}")
            
            if best_match:
                logger.info(f"Fuzzy match found in TM ({best_ratio:.2f}): {text} -> {best_match}")
                return best_match
                
            logger.info(f"No match found in TM for: {text}")
            return None
            
        except Exception as e:
            logger.error(f"Error looking up translation: {str(e)}")
            return None
    
    def in_context_translation(self, text, source_lang, target_lang):
        """
        In-context translation using examples from the TM.
        
        This function would typically use an LLM with examples from the TM.
        For this implementation, we'll skip this step and go directly to the fallback.
        """
        # In a real implementation, this would use an LLM with in-context examples
        logger.info("Skipping in-context translation (not implemented), using fallback")
        return None
    
    def fallback_translation(self, text, source_lang, target_lang):
        """
        Fallback translation using Sarvam API.
        
        Args:
            text (str): The text to translate
            source_lang (str): Source language code (e.g., 'en-IN')
            target_lang (str): Target language code (e.g., 'ta-IN')
            
        Returns:
            str: The translated text
        """
        try:
            logger.info(f"Using Sarvam API for fallback translation: {source_lang} -> {target_lang}")
            
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
            
            response = requests.post(SARVAM_API_ENDPOINT, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result['translated_text']
            
            logger.info(f"Sarvam API translation: {text} -> {translated_text}")
            return translated_text
        
        except Exception as e:
            logger.error(f"Fallback translation failed: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response content: {e.response.content}")
            return f"Translation error: {str(e)}"
    
    def store_translation_triplet(self, english=None, tamil=None, hindi=None, context=None):
        """
        Store a translation triplet in the Translation Memory.
        
        Args:
            english (str): English text
            tamil (str): Tamil text
            hindi (str): Hindi text
            context (str): Optional context information
        """
        # Ensure at least two languages are provided
        if sum(1 for x in [english, tamil, hindi] if x) < 2:
            logger.warning("At least two languages required to store a translation triplet")
            return False
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if we have an existing entry to update
            query_field = next(field for field, value in 
                              [('english', english), ('tamil', tamil), ('hindi', hindi)] 
                              if value is not None)
            query_value = english if query_field == 'english' else tamil if query_field == 'tamil' else hindi
            
            cursor.execute(f"SELECT id, english, tamil, hindi FROM translation_triplets WHERE {query_field} = %s", 
                          (query_value,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing entry
                id, existing_english, existing_tamil, existing_hindi = existing
                
                # Use provided values or keep existing ones
                english = english or existing_english
                tamil = tamil or existing_tamil
                hindi = hindi or existing_hindi
                
                cursor.execute('''
                UPDATE translation_triplets 
                SET english = %s, tamil = %s, hindi = %s, context = %s, timestamp = CURRENT_TIMESTAMP
                WHERE id = %s
                ''', (english, tamil, hindi, context, id))
                
                logger.info(f"Updated translation triplet (ID: {id})")
            else:
                # Insert new entry
                cursor.execute('''
                INSERT INTO translation_triplets (english, tamil, hindi, context)
                VALUES (%s, %s, %s, %s)
                ''', (english, tamil, hindi, context))
                
                logger.info(f"Stored new translation triplet")
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error storing translation triplet: {str(e)}")
            return False
    
    def complete_triplet(self, text, known_lang):
        """
        Complete a translation triplet by translating to all other languages.
        
        Args:
            text (str): The text in a known language
            known_lang (str): The language code of the text
            
        Returns:
            dict: The completed triplet
        """
        # Initialize the triplet with the known text
        triplet = {
            'english': None,
            'tamil': None,
            'hindi': None
        }
        
        known_field = self._language_code_to_field(known_lang)
        triplet[known_field] = text
        
        # Get language codes for the missing languages
        missing_langs = [lang for lang, field in 
                        [('en-IN', 'english'), ('ta-IN', 'tamil'), ('hi-IN', 'hindi')]
                        if field != known_field]
        
        # Translate to missing languages
        for target_lang in missing_langs:
            target_field = self._language_code_to_field(target_lang)
            translation = self.translate(text, known_lang, target_lang)
            triplet[target_field] = translation
        
        # Store the completed triplet
        self.store_translation_triplet(**triplet)
        
        return triplet
    
    def translate(self, text, source_lang, target_lang):
        """
        Translate text from source language to target language.
        
        Args:
            text (str): The text to translate
            source_lang (str): Source language code (e.g., 'en-IN')
            target_lang (str): Target language code (e.g., 'ta-IN')
            
        Returns:
            str: The translated text
        """
        if source_lang == target_lang:
            return text
        
        # 1. Try Translation Memory
        tm_result = self.tm_lookup(text, source_lang, target_lang)
        if tm_result:
            return tm_result
        
        # 2. Try In-Context Translation (skipped in this implementation)
        context_result = self.in_context_translation(text, source_lang, target_lang)
        if context_result:
            # Store the result in the TM before returning
            if source_lang == 'en-IN':
                self.store_translation_triplet(english=text, 
                                              tamil=context_result if target_lang == 'ta-IN' else None,
                                              hindi=context_result if target_lang == 'hi-IN' else None)
            elif source_lang == 'ta-IN':
                self.store_translation_triplet(english=context_result if target_lang == 'en-IN' else None,
                                              tamil=text,
                                              hindi=context_result if target_lang == 'hi-IN' else None)
            elif source_lang == 'hi-IN':
                self.store_translation_triplet(english=context_result if target_lang == 'en-IN' else None,
                                              tamil=context_result if target_lang == 'ta-IN' else None,
                                              hindi=text)
            return context_result
        
        # 3. Use Fallback Translation (Sarvam API)
        fallback_result = self.fallback_translation(text, source_lang, target_lang)
        
        # Store the result in the TM
        if source_lang == 'en-IN':
            if target_lang == 'ta-IN':
                # After getting Tamil, also get Hindi to complete the triplet
                hindi_translation = self.fallback_translation(text, source_lang, 'hi-IN')
                self.store_translation_triplet(english=text, tamil=fallback_result, hindi=hindi_translation)
            else:  # target is Hindi
                # After getting Hindi, also get Tamil to complete the triplet
                tamil_translation = self.fallback_translation(text, source_lang, 'ta-IN')
                self.store_translation_triplet(english=text, tamil=tamil_translation, hindi=fallback_result)
        elif source_lang == 'ta-IN':
            if target_lang == 'en-IN':
                # After getting English, also get Hindi to complete the triplet
                hindi_translation = self.fallback_translation(text, source_lang, 'hi-IN')
                self.store_translation_triplet(english=fallback_result, tamil=text, hindi=hindi_translation)
            else:  # target is Hindi
                # After getting Hindi, also get English to complete the triplet
                english_translation = self.fallback_translation(text, source_lang, 'en-IN')
                self.store_translation_triplet(english=english_translation, tamil=text, hindi=fallback_result)
        elif source_lang == 'hi-IN':
            if target_lang == 'en-IN':
                # After getting English, also get Tamil to complete the triplet
                tamil_translation = self.fallback_translation(text, source_lang, 'ta-IN')
                self.store_translation_triplet(english=fallback_result, tamil=tamil_translation, hindi=text)
            else:  # target is Tamil
                # After getting Tamil, also get English to complete the triplet
                english_translation = self.fallback_translation(text, source_lang, 'en-IN')
                self.store_translation_triplet(english=english_translation, tamil=fallback_result, hindi=text)
        
        return fallback_result
    
    def get_all_triplets(self, limit=100):
        """
        Get all translation triplets from the database.
        
        Args:
            limit (int): Maximum number of triplets to return
            
        Returns:
            list: List of translation triplets
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, english, tamil, hindi, context, timestamp
            FROM translation_triplets
            ORDER BY timestamp DESC
            LIMIT %s
            ''', (limit,))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convert to list of dictionaries
            triplets = []
            for row in results:
                triplet = {
                    'id': row[0],
                    'english': row[1],
                    'tamil': row[2],
                    'hindi': row[3],
                    'context': row[4],
                    'timestamp': row[5]
                }
                triplets.append(triplet)
            
            return triplets
            
        except Exception as e:
            logger.error(f"Error getting triplets: {str(e)}")
            return []

# Create a simple CLI for demonstration
def main():
    translator = PostgresTranslationSystem()
    
    print("\n🌏 PostgreSQL Translation System 🌏")
    print("Supported languages: English, Tamil, Hindi")
    
    while True:
        print("\n" + "="*50)
        print("1. Translate text")
        print("2. View stored translations")
        print("3. Complete translation triplet")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            text = input("\nEnter text to translate: ")
            
            print("\nSelect source language:")
            print("1. English (en-IN)")
            print("2. Tamil (ta-IN)")
            print("3. Hindi (hi-IN)")
            src_choice = input("Enter choice (1-3): ")
            
            source_lang = {
                '1': 'en-IN',
                '2': 'ta-IN',
                '3': 'hi-IN'
            }.get(src_choice, 'en-IN')
            
            print("\nSelect target language:")
            print("1. English (en-IN)")
            print("2. Tamil (ta-IN)")
            print("3. Hindi (hi-IN)")
            tgt_choice = input("Enter choice (1-3): ")
            
            target_lang = {
                '1': 'en-IN',
                '2': 'ta-IN',
                '3': 'hi-IN'
            }.get(tgt_choice, 'ta-IN')
            
            if source_lang == target_lang:
                print("\n⚠️ Source and target languages must be different.")
                continue
            
            print(f"\nTranslating from {translator.supported_languages[source_lang]} to {translator.supported_languages[target_lang]}...")
            translation = translator.translate(text, source_lang, target_lang)
            
            print("\n📝 Translation Result:")
            print(f"{translator.supported_languages[source_lang]}: {text}")
            print(f"{translator.supported_languages[target_lang]}: {translation}")
            
        elif choice == '2':
            triplets = translator.get_all_triplets()
            
            if not triplets:
                print("\nNo translations stored yet.")
                continue
            
            print(f"\n📚 Stored Translations ({len(triplets)} triplets):")
            
            for i, triplet in enumerate(triplets, 1):
                print(f"\n{i}. Triplet ID: {triplet['id']} (Added: {triplet['timestamp']})")
                print(f"   English: {triplet['english'] or 'N/A'}")
                print(f"   Tamil: {triplet['tamil'] or 'N/A'}")
                print(f"   Hindi: {triplet['hindi'] or 'N/A'}")
                if triplet['context']:
                    print(f"   Context: {triplet['context']}")
            
        elif choice == '3':
            text = input("\nEnter text: ")
            
            print("\nSelect text language:")
            print("1. English (en-IN)")
            print("2. Tamil (ta-IN)")
            print("3. Hindi (hi-IN)")
            lang_choice = input("Enter choice (1-3): ")
            
            known_lang = {
                '1': 'en-IN',
                '2': 'ta-IN',
                '3': 'hi-IN'
            }.get(lang_choice, 'en-IN')
            
            print(f"\nCompleting triplet for {text} in {translator.supported_languages[known_lang]}...")
            triplet = translator.complete_triplet(text, known_lang)
            
            print("\n📝 Completed Triplet:")
            print(f"English: {triplet['english']}")
            print(f"Tamil: {triplet['tamil']}")
            print(f"Hindi: {triplet['hindi']}")
            
        elif choice == '4':
            print("\nThank you for using the PostgreSQL Translation System!")
            break
        
        else:
            print("\n⚠️ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 