#!/usr/bin/env python3
"""
Quick verification script to demonstrate export functionality
"""

import pandas as pd
import os

def verify_exports():
    """Verify that the export files contain the expected data"""
    
    print("🔍 TRANSLATION MEMORY EXPORT VERIFICATION")
    print("=" * 50)
    
    # Check if the main export file exists
    if os.path.exists('test_full_export.csv'):
        df = pd.read_csv('test_full_export.csv')
        
        print(f"✅ Full Export File: test_full_export.csv")
        print(f"   Total rows: {len(df)}")
        print(f"   File size: {os.path.getsize('test_full_export.csv'):,} bytes")
        
        # Statistics
        complete_triplets = len(df[df['Entry_Type'] == 'Complete_Triplet'])
        english_entries = len(df[df['Has_English'] == True])
        tamil_entries = len(df[df['Has_Tamil'] == True]) 
        hindi_entries = len(df[df['Has_Hindi'] == True])
        
        print(f"\n📊 Content Statistics:")
        print(f"   Complete triplets: {complete_triplets}")
        print(f"   English entries: {english_entries}")
        print(f"   Tamil entries: {tamil_entries}")
        print(f"   Hindi entries: {hindi_entries}")
        
        # Show sample complete triplets
        print(f"\n🎯 Sample Complete Triplets:")
        sample_triplets = df[df['Entry_Type'] == 'Complete_Triplet'].head(3)
        for idx, row in sample_triplets.iterrows():
            print(f"   {row['ID']:3}: {row['English'][:20]:<20} | {row['Tamil'][:20]:<20} | {row['Hindi'][:20]:<20}")
        
        # Show columns
        print(f"\n📋 Available Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"   - {col}")
            
    else:
        print("❌ Full export file not found. Run: python export_translation_memory.py")
    
    print("\n" + "=" * 50)
    print("✅ Export verification complete!")
    print("\n🚀 Usage Examples:")
    print("   Web Interface: http://localhost:8501")
    print("   Command Line: python export_translation_memory.py --format all")
    print("   Help: python export_translation_memory.py --help")

if __name__ == "__main__":
    verify_exports() 