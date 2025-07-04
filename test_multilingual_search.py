"""
Test script for multilingual search functionality.

This script tests the ability to search for translations across multiple languages
and demonstrates how the semantic search works in a multilingual context.
"""

from advanced_translation_system import AdvancedTranslationSystem
import pandas as pd

def main():
    print("Initializing AdvancedTranslationSystem...")
    translator = AdvancedTranslationSystem()
    
    # Add some test data to ensure we have matching translations
    test_data = [
        {
            'english': 'email settings',
            'tamil': 'மின்னஞ்சல் அமைப்புகள்',
            'hindi': 'ईमेल सेटिंग्स',
            'context': 'Settings menu'
        },
        {
            'english': 'attachment download failed',
            'tamil': 'இணைப்பு பதிவிறக்கம் தோல்வியடைந்தது',
            'hindi': 'अटैचमेंट डाउनलोड विफल',
            'context': 'Error message'
        },
        {
            'english': 'high priority message',
            'tamil': 'உயர் முன்னுரிமை செய்தி',
            'hindi': 'उच्च प्राथमिकता संदेश',
            'context': 'Message priority'
        }
    ]
    
    # Store test data in translation memory
    for entry in test_data:
        translator.store_translation_triplet(**entry)
    
    print("\n=== Testing Multilingual Search ===\n")
    
    # Test searches in different languages
    search_tests = [
        # English queries
        {'query': 'email', 'lang': 'en-IN', 'desc': 'English query for "email"'},
        {'query': 'attachment', 'lang': 'en-IN', 'desc': 'English query for "attachment"'},
        {'query': 'priority', 'lang': 'en-IN', 'desc': 'English query for "priority"'},
        
        # Tamil queries
        {'query': 'மின்னஞ்சல்', 'lang': 'ta-IN', 'desc': 'Tamil query for "email"'},
        {'query': 'இணைப்பு', 'lang': 'ta-IN', 'desc': 'Tamil query for "attachment"'},
        {'query': 'முன்னுரிமை', 'lang': 'ta-IN', 'desc': 'Tamil query for "priority"'},
        
        # Hindi queries
        {'query': 'ईमेल', 'lang': 'hi-IN', 'desc': 'Hindi query for "email"'},
        {'query': 'अटैचमेंट', 'lang': 'hi-IN', 'desc': 'Hindi query for "attachment"'},
        {'query': 'प्राथमिकता', 'lang': 'hi-IN', 'desc': 'Hindi query for "priority"'}
    ]
    
    for test in search_tests:
        query = test['query']
        lang = test['lang']
        desc = test['desc']
        
        print(f"\nTest: {desc}")
        print(f"Query: '{query}' (Language: {lang})")
        
        # Perform database search
        field_name = translator.get_field_name(lang)
        conn = translator.get_db_connection()
        cursor = conn.cursor()
        
        query_str = f"SELECT id, english, tamil, hindi, context FROM translations WHERE {field_name} LIKE ?"
        cursor.execute(query_str, (f"%{query}%",))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Display results
        if rows:
            print(f"Found {len(rows)} results:")
            for row in rows:
                id, en, ta, hi, ctx = row
                print(f"  ID: {id}")
                print(f"  English: {en}")
                print(f"  Tamil: {ta}")
                print(f"  Hindi: {hi}")
                print(f"  Context: {ctx}")
                print("")
        else:
            print("No exact matches found in database.")
            
        # Now try semantic search
        print("Semantic search results:")
        semantic_results = []
        
        # Search in each target language
        for target_lang in ['en-IN', 'ta-IN', 'hi-IN']:
            if target_lang != lang:  # Don't search in the source language
                match, score = translator.search_semantic_similar(query, lang, target_lang)
                if match and score > 0.5:
                    if isinstance(match, dict):
                        # If we got a dictionary with all fields
                        semantic_results.append({
                            'source_text': query,
                            'source_lang': lang,
                            'target_lang': target_lang,
                            'match': match.get(translator.get_field_name(target_lang)),
                            'score': score,
                            'english': match.get('english'),
                            'tamil': match.get('tamil'),
                            'hindi': match.get('hindi')
                        })
                    else:
                        # If we got just the translated text
                        semantic_results.append({
                            'source_text': query,
                            'source_lang': lang,
                            'target_lang': target_lang,
                            'match': match,
                            'score': score
                        })
        
        # Sort by score (highest first)
        semantic_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        if semantic_results:
            for i, result in enumerate(semantic_results):
                print(f"  Result {i+1}: Score {result.get('score', 0):.4f}")
                if 'english' in result:
                    print(f"  English: {result.get('english')}")
                if 'tamil' in result:
                    print(f"  Tamil: {result.get('tamil')}")
                if 'hindi' in result:
                    print(f"  Hindi: {result.get('hindi')}")
                else:
                    print(f"  {result.get('target_lang')}: {result.get('match')}")
                print("")
        else:
            print("  No semantic matches found.")
    
    print("\n=== Testing Cross-Language Translation Path ===\n")
    
    # Test finding a translation path when direct translation is not available
    # Example: If we have A→B and B→C but not A→C, can we find A→C via B?
    
    # First, add a test case with missing translation
    translator.store_translation_triplet(
        english="file compression settings",
        tamil="கோப்பு சுருக்க அமைப்புகள்",
        hindi=None,  # Intentionally missing
        context="Advanced settings"
    )
    
    # Try to translate from English to Hindi
    source_text = "file compression settings"
    source_lang = "en-IN"
    target_lang = "hi-IN"
    
    print(f"Testing translation path for: '{source_text}' ({source_lang} → {target_lang})")
    print(f"Note: Direct translation is intentionally missing")
    
    # First check direct translation
    direct_match, direct_score = translator.search_semantic_similar(source_text, source_lang, target_lang)
    
    if direct_match and direct_score > 0.5:
        print(f"Direct translation found (unexpected):")
        print(f"  Match: {direct_match}")
        print(f"  Score: {direct_score:.4f}")
    else:
        print("Direct translation not found (expected)")
        
        # Try translation via Tamil (bridging language)
        bridge_lang = "ta-IN"
        print(f"Attempting translation via {bridge_lang}")
        
        # Step 1: Translate from source to bridge
        bridge_match, bridge_score = translator.search_semantic_similar(source_text, source_lang, bridge_lang)
        
        if bridge_match and bridge_score > 0.5:
            print(f"First hop successful: {source_lang} → {bridge_lang}")
            print(f"  Intermediate: {bridge_match}")
            print(f"  Score: {bridge_score:.4f}")
            
            # Step 2: Use the bridge text to translate to target
            if isinstance(bridge_match, dict):
                bridge_text = bridge_match.get(translator.get_field_name(bridge_lang))
            else:
                bridge_text = bridge_match
                
            target_match, target_score = translator.search_semantic_similar(bridge_text, bridge_lang, target_lang)
            
            if target_match and target_score > 0.5:
                print(f"Second hop successful: {bridge_lang} → {target_lang}")
                print(f"  Final result: {target_match}")
                print(f"  Score: {target_score:.4f}")
                print(f"Complete path: {source_lang} → {bridge_lang} → {target_lang}")
            else:
                print(f"Second hop failed: {bridge_lang} → {target_lang}")
        else:
            print(f"First hop failed: {source_lang} → {bridge_lang}")
    
    print("\nMultilingual search tests completed!")

if __name__ == "__main__":
    main() 