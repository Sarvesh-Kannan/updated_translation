import sqlite3
from advanced_translation_system import AdvancedTranslationSystem, decode_unicode

def test_unicode_decoding():
    """Test that Unicode escape sequences are properly decoded."""
    # Example of Tamil text with Unicode escapes
    tamil_unicode = r"\u0BAE\u0BC7\u0BB2\u0BCD\u0BC7\u0BB1\u0BCD\u0BB1\u0BAE\u0BCD \u0BA8\u0B9F\u0BC8\u0BAA\u0BC7\u0BB1\u0BC1\u0BA4\u0BB2\u0BCD"
    hindi_unicode = r"\u0909\u091A\u094D\u091A \u092A\u094D\u0930\u093E\u0925\u092E\u093F\u0915\u0924\u093E"
    
    # Test the decode function
    print("Original Tamil Unicode:", tamil_unicode)
    print("Decoded Tamil:", decode_unicode(tamil_unicode))
    print()
    print("Original Hindi Unicode:", hindi_unicode)
    print("Decoded Hindi:", decode_unicode(hindi_unicode))
    print()
    
    # Test retrieving from database and decoding
    conn = sqlite3.connect('translation_memory.db')
    cursor = conn.cursor()
    
    # Sample queries
    cursor.execute("SELECT english, tamil, hindi FROM translations WHERE english = 'upload progress' LIMIT 1")
    row = cursor.fetchone()
    if row:
        print("Example from database - 'upload progress':")
        print("English:", row[0])
        print("Tamil (raw):", row[1])
        print("Tamil (decoded):", decode_unicode(row[1]))
        print("Hindi (raw):", row[2])
        print("Hindi (decoded):", decode_unicode(row[2]))
    
    cursor.execute("SELECT english, tamil, hindi FROM translations WHERE english = 'Inbox' LIMIT 1")
    row = cursor.fetchone()
    if row:
        print("\nExample from database - 'Inbox':")
        print("English:", row[0])
        print("Tamil (raw):", row[1])
        print("Tamil (decoded):", decode_unicode(row[1]))
        print("Hindi (raw):", row[2])
        print("Hindi (decoded):", decode_unicode(row[2]))
    
    conn.close()
    
    # Now test the translation system
    print("\nTesting translation system:")
    translator = AdvancedTranslationSystem()
    
    # Test a few translations
    for text in ["upload progress", "Inbox", "Important"]:
        for target_lang in ["ta-IN", "hi-IN"]:
            translation = translator.translate(text, "en-IN", target_lang)
            print(f"'{text}' -> {target_lang}: '{translation}'")

if __name__ == "__main__":
    test_unicode_decoding() 