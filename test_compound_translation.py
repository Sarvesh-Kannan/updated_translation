import logging
from advanced_translation_system import AdvancedTranslationSystem, decode_unicode

# Configure logging to show detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_compound_translation():
    """Test the compound phrase translation feature of the advanced translation system."""
    
    print("\n" + "="*80)
    print("COMPOUND PHRASE TRANSLATION TEST")
    print("="*80)
    
    # Initialize the advanced translation system
    translator = AdvancedTranslationSystem()
    
    # Test cases with compound phrases
    test_cases = [
        # Simple phrases that should have direct translations
        ("high priority", "en-IN", "ta-IN"),
        ("high priority attachments", "en-IN", "ta-IN"),
        ("please submit attachments", "en-IN", "ta-IN"),
        
        # Test cases for improved handling of common words
        ("this is a test", "en-IN", "ta-IN"),
        ("the document is ready", "en-IN", "ta-IN"),
        
        # Edge case: sentence containing a word that matches a longer phrase in the database
        ("the priority of this task", "en-IN", "ta-IN"),
        
        # Test Hindi translation as well
        ("high priority attachments", "en-IN", "hi-IN"),
    ]
    
    for text, source_lang, target_lang in test_cases:
        print(f"\nTranslating: '{text}' from {source_lang} to {target_lang}")
        translation = translator.translate(text, source_lang, target_lang)
        print(f"Result: '{translation}'")
        
    # Test specific edge case that had problems
    print("\n" + "-"*60)
    print("EDGE CASE: DOCUMENT WORD TEST")
    print("-"*60)
    
    # Store a full sentence with the word "document" in it
    full_sentence = "Please submit the document for review"
    full_sentence_tamil = "மதிப்பாய்வுக்காக ஆவணத்தைச் சமர்ப்பிக்கவும்"
    
    # Store the full sentence translation
    translator.store_translation_triplet(
        english=full_sentence,
        tamil=full_sentence_tamil
    )
    
    # Store a single word translation
    single_word = "document"
    single_word_tamil = "ஆவணம்"
    
    # Store the single word translation
    translator.store_translation_triplet(
        english=single_word,
        tamil=single_word_tamil
    )
    
    # Test a sentence with the word "document" in it
    test_text = "The document needs to be signed"
    print(f"\nTranslating: '{test_text}' from en-IN to ta-IN")
    translation = translator.translate(test_text, "en-IN", "ta-IN")
    print(f"Result: '{translation}'")
    
    # Check if the proper word translation is used, not the full sentence
    if "ஆவணம்" in translation and "மதிப்பாய்வுக்காக ஆவணத்தைச்" not in translation:
        print("✅ Correct: Used proper word translation and avoided full sentence")
    else:
        print("❌ Issue: Did not correctly translate individual word")
        print(f"Expected to find 'ஆவணம்' in the translation")
        print(f"Expected NOT to find 'மதிப்பாய்வுக்காக ஆவணத்தைச்' in the translation")

if __name__ == "__main__":
    test_compound_translation() 