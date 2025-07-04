import sqlite3
import pandas as pd

def view_translations():
    conn = sqlite3.connect('translation_memory.db')
    
    # Get all translations
    query = '''
    SELECT source_text, source_lang, target_text, target_lang, timestamp 
    FROM translations
    ORDER BY timestamp DESC
    '''
    
    # Read into pandas DataFrame for better display
    df = pd.read_sql_query(query, conn)
    
    # Display statistics
    print("\n=== Translation Memory Statistics ===")
    print(f"Total translations: {len(df)}")
    print("\nTranslations by language pair:")
    lang_pairs = df.groupby(['source_lang', 'target_lang']).size()
    print(lang_pairs)
    
    # Display some sample translations
    print("\n=== Sample Translations ===")
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    print(df)
    
    conn.close()

if __name__ == "__main__":
    view_translations() 