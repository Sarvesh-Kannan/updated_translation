import re
import os
from pathlib import Path
from translation_system import MinimalistTranslationSystem

def parse_properties_file(filepath):
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
                # Properties files already have unicode values
                # No need to decode, just use the value as is
                translations[key.strip()] = value
    
    return translations

def load_properties_into_tm():
    """Load translations from properties files into the translation memory."""
    translator = MinimalistTranslationSystem()
    
    # Define the language codes for our files
    files_and_langs = {
        'MessageResources_en.properties': 'en-IN',
        'MessageResources_hi.properties': 'hi-IN',
        'MessageResources_ta.properties': 'ta-IN'
    }
    
    # Parse each properties file
    translations = {}
    for filename, lang_code in files_and_langs.items():
        if os.path.exists(filename):
            translations[lang_code] = parse_properties_file(filename)
            print(f"Loaded {len(translations[lang_code])} translations from {filename}")
        else:
            print(f"Warning: File {filename} not found")
            translations[lang_code] = {}
    
    # Find common keys across all files
    available_langs = list(translations.keys())
    if not available_langs:
        print("No translation files found!")
        return
    
    # Get all keys from the first language
    all_keys = set(translations[available_langs[0]].keys())
    
    # Store triplets in the database
    triplets_added = 0
    
    for key in all_keys:
        # Initialize triplet data
        triplet = {
            'english': None,
            'tamil': None,
            'hindi': None,
            'context': key  # Use the property key as context
        }
        
        # Collect translations for this key from available languages
        for lang in available_langs:
            if key in translations[lang]:
                if lang == 'en-IN':
                    triplet['english'] = translations[lang][key]
                elif lang == 'ta-IN':
                    triplet['tamil'] = translations[lang][key]
                elif lang == 'hi-IN':
                    triplet['hindi'] = translations[lang][key]
        
        # Store if we have at least two languages
        if sum(1 for val in [triplet['english'], triplet['tamil'], triplet['hindi']] if val) >= 2:
            translator.store_translation_triplet(**triplet)
            triplets_added += 1
    
    print(f"Added {triplets_added} translation triplets to the database")

if __name__ == "__main__":
    load_properties_into_tm()
    print("Properties files loaded into translation memory successfully!") 