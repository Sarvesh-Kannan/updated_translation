"""
Test script to verify that semantic search works correctly for all three languages.
"""

from advanced_translation_system import AdvancedTranslationSystem

def main():
    print("Initializing AdvancedTranslationSystem...")
    translator = AdvancedTranslationSystem()
    
    # Test data for semantic search
    test_cases = [
        # English source
        {
            'text': 'Hello world',
            'source_lang': 'en-IN',
            'target_lang': 'ta-IN'
        },
        {
            'text': 'Document viewer',
            'source_lang': 'en-IN',
            'target_lang': 'hi-IN'
        },
        
        # Tamil source
        {
            'text': 'வணக்கம் உலகம்',
            'source_lang': 'ta-IN',
            'target_lang': 'en-IN'
        },
        {
            'text': 'ஆவணம் காட்டி',
            'source_lang': 'ta-IN',
            'target_lang': 'hi-IN'
        },
        
        # Hindi source
        {
            'text': 'नमस्ते दुनिया',
            'source_lang': 'hi-IN',
            'target_lang': 'en-IN'
        },
        {
            'text': 'दस्तावेज़ व्यूअर',
            'source_lang': 'hi-IN',
            'target_lang': 'ta-IN'
        }
    ]
    
    print("\n=== Testing Semantic Search for All Languages ===\n")
    
    for i, case in enumerate(test_cases):
        text = case['text']
        source_lang = case['source_lang']
        target_lang = case['target_lang']
        
        print(f"Test {i+1}: '{text}' ({source_lang}) -> ({target_lang})")
        
        # Test semantic search directly
        semantic_match, score = translator.search_semantic_similar(text, source_lang, target_lang)
        
        if semantic_match:
            print(f"  ✅ Semantic search successful!")
            print(f"  Match: '{semantic_match}'")
            print(f"  Score: {score:.4f}")
        else:
            print(f"  ❌ No semantic match found")
            print(f"  Score: {score:.4f}")
        
        # Now test with translate function which should use semantic search
        translation = translator.translate(text, source_lang, target_lang)
        print(f"  Translation: '{translation}'")
        print("")
    
    print("\n=== Testing Similar Text Searches ===\n")
    
    # Test cases for similar but not exact matches
    similar_test_cases = [
        # English source - slightly different phrasings
        {
            'text': 'Hello beautiful world',  # Similar to "Hello world"
            'source_lang': 'en-IN',
            'target_lang': 'ta-IN'
        },
        {
            'text': 'Document browser',  # Similar to "Document viewer"
            'source_lang': 'en-IN',
            'target_lang': 'hi-IN'
        },
        
        # Tamil source
        {
            'text': 'வணக்கம் அழகான உலகம்',  # Similar to "வணக்கம் உலகம்"
            'source_lang': 'ta-IN',
            'target_lang': 'en-IN'
        },
        
        # Hindi source
        {
            'text': 'नमस्ते सुंदर दुनिया',  # Similar to "नमस्ते दुनिया"
            'source_lang': 'hi-IN',
            'target_lang': 'en-IN'
        }
    ]
    
    for i, case in enumerate(similar_test_cases):
        text = case['text']
        source_lang = case['source_lang']
        target_lang = case['target_lang']
        
        print(f"Similar Test {i+1}: '{text}' ({source_lang}) -> ({target_lang})")
        
        # Test semantic search directly
        semantic_match, score = translator.search_semantic_similar(text, source_lang, target_lang)
        
        if semantic_match:
            print(f"  ✅ Semantic search successful!")
            print(f"  Match: '{semantic_match}'")
            print(f"  Score: {score:.4f}")
        else:
            print(f"  ❌ No semantic match found")
            print(f"  Score: {score:.4f}")
        
        # Now test with translate function
        translation = translator.translate(text, source_lang, target_lang)
        print(f"  Translation: '{translation}'")
        print("")
    
    print("Semantic search tests completed!")

if __name__ == "__main__":
    main() 