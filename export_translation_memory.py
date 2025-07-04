#!/usr/bin/env python3
"""
Translation Memory Export Tool

This script exports all translation data from the SQLite database to various CSV formats.
It provides comprehensive export options for analysis and backup purposes.

Usage:
    python export_translation_memory.py [--format all|complete|english|tamil|hindi] [--output filename.csv]

Author: Translation System
"""

import argparse
import sqlite3
import pandas as pd
import sys
import os
from datetime import datetime
from pathlib import Path

def decode_unicode(text):
    """Decode Unicode escape sequences in text."""
    if not text or not isinstance(text, str):
        return text
    
    try:
        # Check if the text contains Unicode escape sequences
        if '\\u' in text:
            # Use a safer approach to handle potentially malformed Unicode
            result = ''
            i = 0
            while i < len(text):
                if text[i:i+2] == '\\u' and i + 6 <= len(text):
                    try:
                        # Extract 4 hex digits and convert to Unicode character
                        hex_val = int(text[i+2:i+6], 16)
                        result += chr(hex_val)
                        i += 6
                    except ValueError:
                        # If not a valid hex sequence, treat as regular characters
                        result += text[i]
                        i += 1
                else:
                    result += text[i]
                    i += 1
            return result
        return text
    except Exception as e:
        print(f"Error decoding Unicode: {str(e)}")
        return text

class TranslationMemoryExporter:
    """Export translation memory data to various formats"""
    
    def __init__(self, db_path='translation_memory.db'):
        """Initialize the exporter with database path"""
        self.db_path = db_path
        
        # Check if database exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
    
    def get_all_translations(self):
        """Get all translation triplets from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all translations with metadata
            cursor.execute("""
                SELECT id, english, tamil, hindi, context, timestamp 
                FROM translations 
                ORDER BY id
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            # Process the data
            translations = []
            for row in rows:
                translations.append({
                    'ID': row[0],
                    'English': row[1] or '',
                    'Tamil': decode_unicode(row[2]) or '',
                    'Hindi': decode_unicode(row[3]) or '',
                    'Context': row[4] or '',
                    'Timestamp': row[5] or '',
                    'Has_English': bool(row[1] and row[1].strip()),
                    'Has_Tamil': bool(row[2] and row[2].strip()),
                    'Has_Hindi': bool(row[3] and row[3].strip()),
                })
            
            # Add computed fields
            for translation in translations:
                translation['Languages_Count'] = sum([
                    translation['Has_English'],
                    translation['Has_Tamil'],
                    translation['Has_Hindi']
                ])
                translation['Entry_Type'] = (
                    'Complete_Triplet' if all([
                        translation['Has_English'],
                        translation['Has_Tamil'],
                        translation['Has_Hindi']
                    ]) else 'Partial_Entry'
                )
                translation['English_Length'] = len(translation['English'])
                translation['Tamil_Length'] = len(translation['Tamil'])
                translation['Hindi_Length'] = len(translation['Hindi'])
            
            return translations
            
        except Exception as e:
            print(f"Error reading database: {str(e)}")
            return []
    
    def export_all(self, output_file=None):
        """Export all translations to CSV"""
        translations = self.get_all_translations()
        
        if not translations:
            print("No translations found in database.")
            return False
        
        df = pd.DataFrame(translations)
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"complete_translation_memory_{timestamp}.csv"
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Exported {len(translations)} translations to {output_file}")
        return True
    
    def export_complete_triplets(self, output_file=None):
        """Export only complete triplets (all 3 languages) to CSV"""
        translations = self.get_all_translations()
        
        complete_triplets = [t for t in translations if t['Entry_Type'] == 'Complete_Triplet']
        
        if not complete_triplets:
            print("No complete triplets found in database.")
            return False
        
        df = pd.DataFrame(complete_triplets)
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"complete_triplets_only_{timestamp}.csv"
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Exported {len(complete_triplets)} complete triplets to {output_file}")
        return True
    
    def export_by_language(self, language, output_file=None):
        """Export entries that have a specific language"""
        translations = self.get_all_translations()
        
        # Filter by language
        language_key = f'Has_{language.title()}'
        if language_key not in ['Has_English', 'Has_Tamil', 'Has_Hindi']:
            print(f"Invalid language: {language}. Use 'english', 'tamil', or 'hindi'")
            return False
        
        filtered_translations = [t for t in translations if t[language_key]]
        
        if not filtered_translations:
            print(f"No {language} translations found in database.")
            return False
        
        df = pd.DataFrame(filtered_translations)
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"{language}_entries_{timestamp}.csv"
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Exported {len(filtered_translations)} {language} entries to {output_file}")
        return True
    
    def export_statistics(self, output_file=None):
        """Export translation statistics to CSV"""
        translations = self.get_all_translations()
        
        if not translations:
            print("No translations found in database.")
            return False
        
        # Calculate statistics
        total_entries = len(translations)
        en_count = sum(1 for t in translations if t['Has_English'])
        ta_count = sum(1 for t in translations if t['Has_Tamil'])
        hi_count = sum(1 for t in translations if t['Has_Hindi'])
        complete_count = sum(1 for t in translations if t['Entry_Type'] == 'Complete_Triplet')
        partial_count = total_entries - complete_count
        
        # Language pair statistics
        en_ta_pairs = sum(1 for t in translations if t['Has_English'] and t['Has_Tamil'])
        en_hi_pairs = sum(1 for t in translations if t['Has_English'] and t['Has_Hindi'])
        ta_hi_pairs = sum(1 for t in translations if t['Has_Tamil'] and t['Has_Hindi'])
        
        # Average text lengths
        en_avg_length = sum(t['English_Length'] for t in translations if t['Has_English']) / max(en_count, 1)
        ta_avg_length = sum(t['Tamil_Length'] for t in translations if t['Has_Tamil']) / max(ta_count, 1)
        hi_avg_length = sum(t['Hindi_Length'] for t in translations if t['Has_Hindi']) / max(hi_count, 1)
        
        # Context analysis
        contexts = [t['Context'] for t in translations if t['Context']]
        unique_contexts = len(set(contexts))
        
        stats_data = [
            {'Metric': 'Total Entries', 'Value': total_entries, 'Description': 'Total number of translation entries'},
            {'Metric': 'English Entries', 'Value': en_count, 'Description': 'Entries with English text'},
            {'Metric': 'Tamil Entries', 'Value': ta_count, 'Description': 'Entries with Tamil text'},
            {'Metric': 'Hindi Entries', 'Value': hi_count, 'Description': 'Entries with Hindi text'},
            {'Metric': 'Complete Triplets', 'Value': complete_count, 'Description': 'Entries with all 3 languages'},
            {'Metric': 'Partial Entries', 'Value': partial_count, 'Description': 'Entries missing one or more languages'},
            {'Metric': 'English-Tamil Pairs', 'Value': en_ta_pairs, 'Description': 'Entries with both English and Tamil'},
            {'Metric': 'English-Hindi Pairs', 'Value': en_hi_pairs, 'Description': 'Entries with both English and Hindi'},
            {'Metric': 'Tamil-Hindi Pairs', 'Value': ta_hi_pairs, 'Description': 'Entries with both Tamil and Hindi'},
            {'Metric': 'Avg English Length', 'Value': round(en_avg_length, 2), 'Description': 'Average character length of English text'},
            {'Metric': 'Avg Tamil Length', 'Value': round(ta_avg_length, 2), 'Description': 'Average character length of Tamil text'},
            {'Metric': 'Avg Hindi Length', 'Value': round(hi_avg_length, 2), 'Description': 'Average character length of Hindi text'},
            {'Metric': 'Unique Contexts', 'Value': unique_contexts, 'Description': 'Number of unique context tags'},
        ]
        
        df = pd.DataFrame(stats_data)
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"translation_memory_statistics_{timestamp}.csv"
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Exported translation statistics to {output_file}")
        return True
    
    def print_summary(self):
        """Print a summary of the translation memory"""
        translations = self.get_all_translations()
        
        if not translations:
            print("No translations found in database.")
            return
        
        total_entries = len(translations)
        en_count = sum(1 for t in translations if t['Has_English'])
        ta_count = sum(1 for t in translations if t['Has_Tamil'])
        hi_count = sum(1 for t in translations if t['Has_Hindi'])
        complete_count = sum(1 for t in translations if t['Entry_Type'] == 'Complete_Triplet')
        
        print(f"\n📊 Translation Memory Summary")
        print(f"=" * 50)
        print(f"Total Entries:     {total_entries:,}")
        print(f"English Entries:   {en_count:,}")
        print(f"Tamil Entries:     {ta_count:,}")
        print(f"Hindi Entries:     {hi_count:,}")
        print(f"Complete Triplets: {complete_count:,}")
        print(f"Partial Entries:   {total_entries - complete_count:,}")
        print(f"=" * 50)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Export Translation Memory to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_translation_memory.py                    # Export all translations
    python export_translation_memory.py --format complete  # Export complete triplets only
    python export_translation_memory.py --format english   # Export English entries only
    python export_translation_memory.py --format stats     # Export statistics
    python export_translation_memory.py --output my_data.csv --format all
        """
    )
    
    parser.add_argument(
        '--format', 
        choices=['all', 'complete', 'english', 'tamil', 'hindi', 'stats'],
        default='all',
        help='Export format (default: all)'
    )
    
    parser.add_argument(
        '--output', 
        help='Output CSV file name (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--db', 
        default='translation_memory.db',
        help='Database file path (default: translation_memory.db)'
    )
    
    parser.add_argument(
        '--summary', 
        action='store_true',
        help='Print database summary and exit'
    )
    
    args = parser.parse_args()
    
    try:
        exporter = TranslationMemoryExporter(args.db)
        
        if args.summary:
            exporter.print_summary()
            return
        
        # Export based on format
        success = False
        if args.format == 'all':
            success = exporter.export_all(args.output)
        elif args.format == 'complete':
            success = exporter.export_complete_triplets(args.output)
        elif args.format in ['english', 'tamil', 'hindi']:
            success = exporter.export_by_language(args.format, args.output)
        elif args.format == 'stats':
            success = exporter.export_statistics(args.output)
        
        if success:
            print(f"\n✅ Export completed successfully!")
            exporter.print_summary()
        else:
            print(f"\n❌ Export failed!")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

