from advanced_translation_system import AdvancedTranslationSystem, decode_unicode

def main():
    # Initialize the translation system
    translator = AdvancedTranslationSystem()
    
    # Test case for the fixed "high priority attachments" translation
    print("\n" + "="*80)
    print("VERIFICATION TEST: high priority attachments")
    print("="*80)
    
    # Test the compound phrase
    compound_phrase = "high priority attachments"
    translation = translator.translate(compound_phrase, "en-IN", "ta-IN")
    
    print(f"\nTranslation of '{compound_phrase}' to Tamil:")
    print(f"Result: '{translation}'")
    
    # Verify against expected translation
    expected = "உயர் முன்னுரிமை இணைப்புகள்"
    print(f"\nExpected: '{expected}'")
    print(f"Actual:   '{translation}'")
    
    # Check if the translation is correct
    if translation == expected:
        print("\n✅ PASS: Fixed translation works correctly!")
    else:
        print("\n❌ FAIL: Translation does not match expected result.")
        print("Actual bytes:", [b for b in translation.encode('utf-8')])
        print("Expected bytes:", [b for b in expected.encode('utf-8')])

if __name__ == "__main__":
    main() 