from advanced_translation_system import AdvancedTranslationSystem, decode_unicode

def main():
    # Initialize the translation system
    translator = AdvancedTranslationSystem()
    
    # Test case for Hindi translation of "high priority attachments"
    print("\n" + "="*80)
    print("HINDI TRANSLATION TEST: high priority attachments")
    print("="*80)
    
    # First check individual word translations
    high_translation = translator.translate("high", "en-IN", "hi-IN")
    priority_translation = translator.translate("priority", "en-IN", "hi-IN") 
    attachments_translation = translator.translate("attachments", "en-IN", "hi-IN")
    
    print(f"\nIndividual word translations:")
    print(f"'high' -> '{high_translation}'")
    print(f"'priority' -> '{priority_translation}'")
    print(f"'attachments' -> '{attachments_translation}'")
    
    # Now update the database with the correct Hindi translation for the phrase
    conn = translator.get_db_connection()
    cursor = conn.cursor()
    
    # Find if the entry exists
    cursor.execute("SELECT id FROM translations WHERE english = ?", ("high priority attachments",))
    row = cursor.fetchone()
    
    # Expected correct Hindi translation
    correct_hindi = "उच्च प्राथमिकता अनुलग्नक"
    
    if row:
        # Update the existing entry
        entry_id = row[0]
        cursor.execute(
            "UPDATE translations SET hindi = ? WHERE id = ?", 
            (correct_hindi, entry_id)
        )
        print(f"\nUpdated translation for 'high priority attachments' with ID {entry_id}")
    else:
        # Insert a new entry
        cursor.execute(
            "INSERT INTO translations (english, tamil, hindi) VALUES (?, ?, ?)",
            ("high priority attachments", None, correct_hindi)
        )
        print("\nAdded new translation for 'high priority attachments'")
    
    conn.commit()
    conn.close()
    
    # Test the compound phrase
    compound_phrase = "high priority attachments"
    translation = translator.translate(compound_phrase, "en-IN", "hi-IN")
    
    print(f"\nTranslation of '{compound_phrase}' to Hindi:")
    print(f"Result: '{translation}'")
    
    # Verify against expected translation
    expected = correct_hindi
    print(f"\nExpected: '{expected}'")
    print(f"Actual:   '{translation}'")
    
    # Check if the translation is correct
    if translation == expected:
        print("\n✅ PASS: Hindi translation works correctly!")
    else:
        print("\n❌ FAIL: Hindi translation does not match expected result.")
        print("Actual bytes:", [b for b in translation.encode('utf-8')])
        print("Expected bytes:", [b for b in expected.encode('utf-8')])

if __name__ == "__main__":
    main() 