import streamlit as st
import pandas as pd
import os
import sys
from advanced_translation_system import AdvancedTranslationSystem, decode_unicode

def initialize_translator():
    """Initialize the translation system with error handling."""
    try:
        # Temporarily disable stdout to suppress PyTorch warnings
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # Initialize the system
        translator = AdvancedTranslationSystem()
        
        # Restore stdout
        sys.stdout = old_stdout
        return translator
    except RuntimeError as e:
        # Handle the PyTorch-Streamlit compatibility error
        if "__path__._path" in str(e) and "torch" in str(e):
            # This is the PyTorch path issue - try again with a different approach
            try:
                # Direct initialization without the problematic Streamlit monitoring
                translator = AdvancedTranslationSystem()
                return translator
            except Exception as inner_e:
                st.error(f"Failed to initialize translation system: {str(inner_e)}")
                return None
        else:
            # Some other runtime error
            st.error(f"Runtime error initializing translation system: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error initializing translation system: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Advanced Translation System",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Advanced Translation System")
    st.markdown("""
    ### Context-Aware Multilingual Translation with Semantic Search
    
    This system uses:
    - Sentence embeddings and FAISS for semantic search in English, Tamil, and Hindi
    - SQLite database for translation memory
    - Sarvam API as fallback
    - Dynamic translation memory updates
    """)
    
    # Initialize the translation system
    if 'translator' not in st.session_state:
        with st.spinner("Loading translation system..."):
            st.session_state.translator = initialize_translator()
            
            if st.session_state.translator is None:
                st.error("Failed to initialize the translation system. Please refresh the page and try again.")
                st.stop()
    
    # Layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Translate", "Multilingual Search", "Translation Memory", "System Maintenance"])
    
    with tab1:
        st.subheader("Translation Interface")
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "Source Language",
                options=["en-IN", "ta-IN", "hi-IN"],
                format_func=lambda x: {"en-IN": "English", "ta-IN": "Tamil", "hi-IN": "Hindi"}[x],
                key="source_lang"
            )
            
            source_text = st.text_area("Enter text to translate", height=200, key="source_text")
            
            use_bridging = st.checkbox("Enable translation bridging", 
                                      help="When direct translation isn't available, try using a bridge language")
            
        with col2:
            target_lang = st.selectbox(
                "Target Language",
                options=["en-IN", "ta-IN", "hi-IN"],
                format_func=lambda x: {"en-IN": "English", "ta-IN": "Tamil", "hi-IN": "Hindi"}[x],
                key="target_lang"
            )
            
            translate_button = st.button("Translate", key="translate_button")
            
            if translate_button and source_text:
                with st.spinner("Translating..."):
                    if use_bridging:
                        translation, path = st.session_state.translator.translate_with_bridging(
                            source_text, source_lang, target_lang
                        )
                        
                        # Show the translation path
                        path_str = " → ".join([{"en-IN": "English", "ta-IN": "Tamil", "hi-IN": "Hindi"}[lang] for lang in path])
                        st.info(f"Translation path: {path_str}")
                    else:
                        translation = st.session_state.translator.translate(source_text, source_lang, target_lang)
                    
                    if translation:
                        st.text_area("Translation", value=translation, height=200, key="translation_result")
                        st.success("Translation complete!")
                        
                        # Store in translation memory for future reference
                        try:
                            # Complete the translation triplet
                            result = st.session_state.translator.complete_triplet(source_text, source_lang)
                            st.session_state.translator.store_translation_triplet(
                                english=result.get("english"),
                                tamil=result.get("tamil"),
                                hindi=result.get("hindi"),
                                context="From web interface"
                            )
                        except Exception as e:
                            st.error(f"Error storing translation: {str(e)}")
                    else:
                        st.error("Translation failed!")
    
    with tab2:
        st.subheader("Multilingual Search")
        
        search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
        
        with search_col1:
            ml_search_text = st.text_input("Search across all languages", key="ml_search")
        
        with search_col2:
            ml_search_lang = st.selectbox(
                "Query Language",
                options=["auto", "en-IN", "ta-IN", "hi-IN"],
                format_func=lambda x: {
                    "auto": "Auto-detect", 
                    "en-IN": "English", 
                    "ta-IN": "Tamil", 
                    "hi-IN": "Hindi"
                }[x],
                key="ml_search_lang"
            )
        
        with search_col3:
            ml_threshold = st.slider(
                "Similarity Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05,
                key="ml_threshold"
            )
        
        if ml_search_text:
            with st.spinner("Searching..."):
                try:
                    # Use auto-detect if auto is selected
                    query_lang = None if ml_search_lang == "auto" else ml_search_lang
                    
                    # Perform multilingual search
                    results = st.session_state.translator.multilingual_search(
                        ml_search_text, 
                        query_lang=query_lang, 
                        threshold=ml_threshold,
                        max_results=20
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} results")
                        
                        # Create tabs for different result views
                        result_tab1, result_tab2 = st.tabs(["Table View", "Detailed View"])
                        
                        with result_tab1:
                            # Convert to DataFrame for table view
                            df_data = []
                            for r in results:
                                df_data.append({
                                    "Score": f"{r.get('score', 0):.2f}",
                                    "Match Type": r.get('match_type', ''),
                                    "English": r.get('english', ''),
                                    "Tamil": r.get('tamil', ''),
                                    "Hindi": r.get('hindi', ''),
                                    "Context": r.get('context', '')
                                })
                            
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
                        
                        with result_tab2:
                            # Detailed view with expanders
                            for i, r in enumerate(results):
                                score = r.get('score', 0)
                                match_type = r.get('match_type', '')
                                
                                with st.expander(f"Result {i+1}: Score {score:.2f} ({match_type})"):
                                    cols = st.columns(3)
                                    
                                    with cols[0]:
                                        st.markdown("**English**")
                                        st.text(r.get('english', ''))
                                    
                                    with cols[1]:
                                        st.markdown("**Tamil**")
                                        st.text(r.get('tamil', ''))
                                    
                                    with cols[2]:
                                        st.markdown("**Hindi**")
                                        st.text(r.get('hindi', ''))
                                    
                                    st.markdown(f"**Context:** {r.get('context', '')}")
                                    st.markdown(f"**ID:** {r.get('id', '')}")
                    else:
                        st.info("No matching translations found.")
                        
                except Exception as e:
                    st.error(f"Error performing multilingual search: {str(e)}")
            
    with tab3:
        st.subheader("Translation Memory")
        
        # Search in translation memory
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_text = st.text_input("Search translation memory", key="search_tm")
        
        with search_col2:
            search_lang = st.selectbox(
                "Search in",
                options=["en-IN", "ta-IN", "hi-IN", "all"],
                format_func=lambda x: {"en-IN": "English", "ta-IN": "Tamil", "hi-IN": "Hindi", "all": "All Languages"}[x],
                key="search_lang"
            )
        
        if search_text:
            with st.spinner("Searching..."):
                try:
                    conn = st.session_state.translator.get_db_connection()
                    cursor = conn.cursor()
                    
                    if search_lang == "all":
                        # Search in all languages
                        cursor.execute("""
                            SELECT id, english, tamil, hindi, context FROM translations
                            WHERE english LIKE ? OR tamil LIKE ? OR hindi LIKE ?
                            ORDER BY id DESC LIMIT 50
                        """, (f"%{search_text}%", f"%{search_text}%", f"%{search_text}%"))
                    else:
                        # Search in specific language
                        field = st.session_state.translator.get_field_name(search_lang)
                        cursor.execute(f"""
                            SELECT id, english, tamil, hindi, context FROM translations
                            WHERE {field} LIKE ?
                            ORDER BY id DESC LIMIT 50
                        """, (f"%{search_text}%",))
                    
                    rows = cursor.fetchall()
                    conn.close()
                    
                    if rows:
                        df = pd.DataFrame(rows, columns=["ID", "English", "Tamil", "Hindi", "Context"])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No matching translations found.")
                        
                except Exception as e:
                    st.error(f"Error searching translation memory: {str(e)}")
        
        # Add new translation entry manually
        st.subheader("Add New Translation")
        
        add_col1, add_col2, add_col3 = st.columns(3)
        
        with add_col1:
            new_en = st.text_area("English", key="new_en", height=100)
        
        with add_col2:
            new_ta = st.text_area("Tamil", key="new_ta", height=100)
        
        with add_col3:
            new_hi = st.text_area("Hindi", key="new_hi", height=100)
        
        new_context = st.text_input("Context (optional)", key="new_context")
        
        if st.button("Add Translation", key="add_translation"):
            if new_en or new_ta or new_hi:
                try:
                    st.session_state.translator.store_translation_triplet(
                        english=new_en if new_en else None,
                        tamil=new_ta if new_ta else None,
                        hindi=new_hi if new_hi else None,
                        context=new_context if new_context else "Manually added"
                    )
                    st.success("Translation added successfully!")
                    # Clear input fields
                    st.session_state.new_en = ""
                    st.session_state.new_ta = ""
                    st.session_state.new_hi = ""
                    st.session_state.new_context = ""
                except Exception as e:
                    st.error(f"Error adding translation: {str(e)}")
            else:
                st.warning("At least one language field must be filled.")
    
    with tab4:
        st.subheader("System Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Rebuild Indices", key="rebuild_indices"):
                with st.spinner("Rebuilding indices..."):
                    stats = st.session_state.translator.rebuild_indices()
                st.success(f"Indices rebuilt successfully! English: {stats['english_entries']}, Tamil: {stats['tamil_entries']}, Hindi: {stats['hindi_entries']} entries indexed.")
        
        with col2:
            if st.button("Check & Fix Consistency", key="check_consistency"):
                with st.spinner("Checking and fixing translation consistency..."):
                    stats = st.session_state.translator.rebuild_and_check_consistency()
                st.success(f"Consistency check complete! Found {stats['inconsistencies_found']} inconsistencies, fixed {stats['entries_fixed']} entries. Rebuilt indices with English: {stats['english_entries']}, Tamil: {stats['tamil_entries']}, Hindi: {stats['hindi_entries']} entries.")
        
        # Database statistics
        st.subheader("Database Statistics")
        
        try:
            conn = st.session_state.translator.get_db_connection()
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute("SELECT COUNT(*) FROM translations")
            total_entries = cursor.fetchone()[0]
            
            # English entries
            cursor.execute("SELECT COUNT(*) FROM translations WHERE english IS NOT NULL AND english != ''")
            english_entries = cursor.fetchone()[0]
            
            # Tamil entries
            cursor.execute("SELECT COUNT(*) FROM translations WHERE tamil IS NOT NULL AND tamil != ''")
            tamil_entries = cursor.fetchone()[0]
            
            # Hindi entries
            cursor.execute("SELECT COUNT(*) FROM translations WHERE hindi IS NOT NULL AND hindi != ''")
            hindi_entries = cursor.fetchone()[0]
            
            # Complete triplets
            cursor.execute("""
                SELECT COUNT(*) FROM translations 
                WHERE english IS NOT NULL AND english != '' 
                AND tamil IS NOT NULL AND tamil != '' 
                AND hindi IS NOT NULL AND hindi != ''
            """)
            complete_triplets = cursor.fetchone()[0]
            
            conn.close()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Entries", total_entries)
            col2.metric("English Entries", english_entries)
            col3.metric("Tamil Entries", tamil_entries)
            col4.metric("Hindi Entries", hindi_entries)
            col5.metric("Complete Triplets", complete_triplets)
            
            # Show a percentage of completion for each language pair
            st.subheader("Translation Coverage")
            
            # Calculate percentages
            en_ta_percent = round((english_entries / total_entries) * 100) if total_entries > 0 else 0
            en_hi_percent = round((english_entries / total_entries) * 100) if total_entries > 0 else 0
            ta_hi_percent = round((tamil_entries / total_entries) * 100) if total_entries > 0 else 0
            complete_percent = round((complete_triplets / total_entries) * 100) if total_entries > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.progress(en_ta_percent / 100, f"English-Tamil: {en_ta_percent}%")
            col2.progress(en_hi_percent / 100, f"English-Hindi: {en_hi_percent}%")
            col3.progress(ta_hi_percent / 100, f"Tamil-Hindi: {ta_hi_percent}%")
            col4.progress(complete_percent / 100, f"Complete Triplets: {complete_percent}%")
            
        except Exception as e:
            st.error(f"Error fetching database statistics: {str(e)}")

if __name__ == "__main__":
    main() 