import logging
from advanced_translation_system import AdvancedTranslationSystem, decode_unicode

# Configure logging to show detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_complex_sentence_case():
    """Test the edge case where a word in a sentence might match a full stored sentence."""
    
    print("\n" + "="*80)
    print("TESTING EDGE CASE: WORD IN SENTENCE MATCHES FULL STORED SENTENCE")
    print("="*80)
    
    # Initialize the advanced translation system
    translator = AdvancedTranslationSystem()
    
    # Setup test case
    # We need to first add a test case where a word is part of a longer sentence in the database
    # Let's create this situation for testing
    
    # 1. Add a full sentence to the database
    full_sentence = "Please submit the document for review"
    full_sentence_tamil = "மதிப்பாய்வுக்காக ஆவணத்தைச் சமர்ப்பிக்கவும்"
    
    # Store the full sentence translation
    translator.store_translation_triplet(
        english=full_sentence,
        tamil=full_sentence_tamil
    )
    
    # 2. Add a single word translation
    single_word = "document"
    single_word_tamil = "ஆவணம்"
    
    # Store the single word translation
    translator.store_translation_triplet(
        english=single_word,
        tamil=single_word_tamil
    )
    
    # 3. Now test a new sentence that contains the single word
    test_sentence = "The document needs to be signed"
    
    print(f"\nTranslating: '{test_sentence}'")
    translation = translator.translate(test_sentence, "en-IN", "ta-IN")
    print(f"Result: '{translation}'")
    
    # Verify that we're getting the proper word translation for "document"
    # rather than the full sentence translation containing that word
    expected_word_in_result = single_word_tamil
    print(f"\nChecking if result contains proper word translation: '{expected_word_in_result}'")
    contains_expected = expected_word_in_result in translation
    print(f"Contains proper word translation: {'✅' if contains_expected else '❌'}")
    
    unwanted_full_translation = full_sentence_tamil
    print(f"\nChecking if result contains unwanted full sentence translation: '{unwanted_full_translation}'")
    contains_unwanted = unwanted_full_translation in translation
    print(f"Contains unwanted full sentence: {'❌' if contains_unwanted else '✅'}")
    
    # Test a second case with a slightly different scenario
    print("\n" + "-"*60)
    print("TESTING ANOTHER EDGE CASE: WORD IN MIDDLE OF STORED SENTENCE")
    print("-"*60)
    
    # 4. Add a translation where a common word is in the middle of a sentence
    mid_sentence = "Set priority to high for this task"
    mid_sentence_tamil = "இந்த பணிக்கு முன்னுரிமையை உயர்வாக அமைக்கவும்"
    
    # Store the sentence with the word in the middle
    translator.store_translation_triplet(
        english=mid_sentence,
        tamil=mid_sentence_tamil
    )
    
    # 5. Add the single word translation
    priority_word = "priority"
    priority_word_tamil = "முன்னுரிமை"
    
    translator.store_translation_triplet(
        english=priority_word,
        tamil=priority_word_tamil
    )
    
    # 6. Test a new sentence containing the word
    test_sentence2 = "This is a priority message"
    
    print(f"\nTranslating: '{test_sentence2}'")
    translation2 = translator.translate(test_sentence2, "en-IN", "ta-IN")
    print(f"Result: '{translation2}'")
    
    # Verify that we're getting the proper word translation for "priority"
    # rather than the full sentence translation containing that word
    expected_word_in_result2 = priority_word_tamil
    print(f"\nChecking if result contains proper word translation: '{expected_word_in_result2}'")
    contains_expected2 = expected_word_in_result2 in translation2
    print(f"Contains proper word translation: {'✅' if contains_expected2 else '❌'}")
    
    unwanted_full_translation2 = mid_sentence_tamil
    print(f"\nChecking if result contains unwanted full sentence translation: '{unwanted_full_translation2}'")
    contains_unwanted2 = unwanted_full_translation2 in translation2
    print(f"Contains unwanted full sentence: {'❌' if contains_unwanted2 else '✅'}")

if __name__ == "__main__":
    test_complex_sentence_case() 