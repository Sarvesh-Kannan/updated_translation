from advanced_translation_system import AdvancedTranslationSystem

def rebuild_indices():
    """Rebuild all indices from scratch."""
    try:
        print("Initializing the AdvancedTranslationSystem...")
        translator = AdvancedTranslationSystem()
        
        print("Rebuilding indices...")
        stats = translator.rebuild_indices()
        
        print(f"Rebuilt indices with:")
        print(f"- {stats['english_entries']} English entries")
        print(f"- {stats['tamil_entries']} Tamil entries")
        print(f"- {stats['hindi_entries']} Hindi entries")
        
        # Test translations after rebuild
        test_translations(translator)
        
    except Exception as e:
        print(f"Error rebuilding indices: {e}")

def test_translations(translator):
    """Test a few translations to ensure they work properly."""
    print("\nTesting translations after index rebuild:")
    
    # Test all language pairs
    test_pairs = [
        # Source language, target language, text
        ('en-IN', 'ta-IN', 'Hello world'),
        ('en-IN', 'hi-IN', 'Document'),
        ('ta-IN', 'en-IN', 'வணக்கம் உலகம்'),
        ('ta-IN', 'hi-IN', 'ஆவணம்'),
        ('hi-IN', 'en-IN', 'नमस्ते दुनिया'),
        ('hi-IN', 'ta-IN', 'दस्तावेज़')
    ]
    
    for source_lang, target_lang, text in test_pairs:
        result = translator.translate(text, source_lang, target_lang)
        print(f"{source_lang} → {target_lang}: '{text}' → '{result}'")

if __name__ == "__main__":
    rebuild_indices() 