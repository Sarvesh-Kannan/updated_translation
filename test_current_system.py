#!/usr/bin/env python3
"""
Test script for the current translation system
"""

from simple_translation_test import SimpleTranslationSystem, decode_unicode
import time

def main():
    print("🌐 Testing Advanced Translation System")
    print("=" * 50)
    
    # Initialize the system
    print("📊 Initializing translation system...")
    try:
        translator = SimpleTranslationSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Get system statistics
    print("\n📈 System Statistics:")
    try:
        triplets = translator.get_all_triplets()
        total_entries = len(triplets)
        en_count = sum(1 for t in triplets if t.get('english'))
        hi_count = sum(1 for t in triplets if t.get('hindi'))
        ta_count = sum(1 for t in triplets if t.get('tamil'))
        
        print(f"   Total Entries: {total_entries}")
        print(f"   English Entries: {en_count}")
        print(f"   Hindi Entries: {hi_count}")
        print(f"   Tamil Entries: {ta_count}")
    except Exception as e:
        print(f"   Error getting statistics: {e}")
    
    # Test translations
    print("\n🔄 Testing Translations:")
    print("-" * 30)
    
    test_cases = [
        # Test cases from database
        ("Hello", "en-IN", "hi-IN", "Database lookup"),
        ("Thank you", "en-IN", "ta-IN", "Database lookup"),
        ("Good morning", "en-IN", "hi-IN", "Database lookup"),
        
        # Test cases that might need API
        ("How are you today?", "en-IN", "hi-IN", "API fallback"),
        ("I love programming", "en-IN", "ta-IN", "API fallback"),
        
        # Test reverse translations
        ("नमस्ते", "hi-IN", "en-IN", "Hindi to English"),
        ("வணக்கம்", "ta-IN", "en-IN", "Tamil to English"),
    ]
    
    for i, (text, source_lang, target_lang, description) in enumerate(test_cases, 1):
        print(f"\n{i}. {description}")
        print(f"   Input: '{text}' ({source_lang})")
        
        start_time = time.time()
        try:
            # Perform translation
            translation = translator.translate(text, source_lang, target_lang)
            end_time = time.time()
            
            # Check if it was from database or API
            exact_match, _ = translator.search_exact_match(text, source_lang, target_lang)
            method = "💾 Database" if exact_match else "🌐 Sarvam AI"
            
            print(f"   Output: '{translation}' ({target_lang})")
            print(f"   Method: {method}")
            print(f"   Time: {end_time - start_time:.3f}s")
            
            if translation == text:
                print("   Status: ⚠️ No translation (returned original)")
            else:
                print("   Status: ✅ Success")
                
        except Exception as e:
            print(f"   Status: ❌ Error - {e}")
    
    # Test some specific database entries
    print("\n🔍 Testing Specific Database Entries:")
    print("-" * 40)
    
    # Get a few sample entries from database
    try:
        sample_triplets = triplets[:5]  # Get first 5 entries
        
        for i, triplet in enumerate(sample_triplets, 1):
            english = triplet.get('english', '')
            hindi = decode_unicode(triplet.get('hindi', ''))
            tamil = decode_unicode(triplet.get('tamil', ''))
            
            if english and hindi:
                print(f"\n{i}a. EN->HI Database Test")
                print(f"    Input: '{english}'")
                result = translator.translate(english, 'en-IN', 'hi-IN')
                print(f"    Expected: '{hindi}'")
                print(f"    Got: '{result}'")
                print(f"    Match: {'✅' if result == hindi else '❌'}")
            
            if english and tamil:
                print(f"\n{i}b. EN->TA Database Test")
                print(f"    Input: '{english}'")
                result = translator.translate(english, 'en-IN', 'ta-IN')
                print(f"    Expected: '{tamil}'")
                print(f"    Got: '{result}'")
                print(f"    Match: {'✅' if result == tamil else '❌'}")
                
    except Exception as e:
        print(f"Error testing database entries: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Translation system testing completed!")

if __name__ == "__main__":
    main() 