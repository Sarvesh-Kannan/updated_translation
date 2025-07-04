import time
from advanced_translation_system import AdvancedTranslationSystem

def test_all_translation_directions():
    """Test translations in all possible language direction combinations."""
    translator = AdvancedTranslationSystem()
    
    # Test data in all three languages
    test_data = {
        'en-IN': [
            "Hello world", 
            "Document", 
            "Previous", 
            "Next",
            "Important message"
        ],
        'ta-IN': [
            "வணக்கம் உலகம்", 
            "ஆவணம்", 
            "முந்தைய", 
            "அடுத்து",
            "முக்கியமான செய்தி"
        ],
        'hi-IN': [
            "नमस्ते दुनिया", 
            "दस्तावेज़", 
            "पिछला", 
            "अगला",
            "महत्वपूर्ण संदेश"
        ]
    }
    
    languages = list(translator.supported_languages.keys())
    
    print("\n===== TESTING ALL TRANSLATION DIRECTIONS =====")
    
    # Test all language direction combinations
    for source_lang in languages:
        for target_lang in languages:
            if source_lang == target_lang:
                continue
                
            print(f"\n\n=== Testing {source_lang} → {target_lang} ===")
            
            for text in test_data[source_lang]:
                # Translate
                start_time = time.time()
                translation = translator.translate(text, source_lang, target_lang)
                end_time = time.time()
                
                # Print results
                print(f"Source ({source_lang}): {text}")
                print(f"Target ({target_lang}): {translation}")
                print(f"Time: {(end_time - start_time):.2f} seconds")
                print("---")
                
                # Check if translation is empty or same as source
                if not translation or translation == text:
                    print("⚠️ WARNING: Translation failed or returned source text")
                
                # Small delay to avoid overloading
                time.sleep(0.5)
    
    print("\n===== TESTING SEMANTIC SEARCH DIRECTLY =====")
    
    # Test semantic search specifically for all language directions
    for source_lang in languages:
        for target_lang in languages:
            if source_lang == target_lang:
                continue
                
            print(f"\n=== Testing semantic search {source_lang} → {target_lang} ===")
            
            for text in test_data[source_lang]:
                semantic_match, score = translator.search_semantic_similar(text, source_lang)
                print(f"Query ({source_lang}): {text}")
                
                if semantic_match:
                    source_field = translator.get_field_name(source_lang)
                    target_field = translator.get_field_name(target_lang)
                    print(f"Found match with score: {score:.2f}")
                    print(f"Source match: {semantic_match.get(source_field)}")
                    print(f"Target match: {semantic_match.get(target_field)}")
                else:
                    print("No semantic match found")
                print("---")
                
                # Small delay to avoid overloading
                time.sleep(0.2)

if __name__ == "__main__":
    test_all_translation_directions() 