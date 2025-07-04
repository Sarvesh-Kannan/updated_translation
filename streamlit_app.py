import streamlit as st
import pandas as pd
import time
import logging
from simple_translation_test import SimpleTranslationSystem, decode_unicode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Translation System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .stat-box {
        text-align: center;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_translation_system():
    """Load the translation system with caching"""
    try:
        system = SimpleTranslationSystem()
        return system
    except Exception as e:
        st.error(f"Error loading translation system: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌐 Advanced Translation System</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Sarvam AI & Local Translation Memory**")
    
    # Load translation system
    translator = load_translation_system()
    
    if translator is None:
        st.error("Failed to load translation system. Please check the configuration.")
        return
    
    # Sidebar for system information
    with st.sidebar:
        st.header("📊 System Information")
        
        # Get database statistics
        try:
            triplets = translator.get_all_triplets()
            total_entries = len(triplets)
            en_count = sum(1 for t in triplets if t['english'])
            ta_count = sum(1 for t in triplets if t['tamil'])
            hi_count = sum(1 for t in triplets if t['hindi'])
            
            st.metric("Total Entries", total_entries)
            st.metric("English Entries", en_count)
            st.metric("Tamil Entries", ta_count)
            st.metric("Hindi Entries", hi_count)
            
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 🔧 Features")
        st.markdown("- ✅ Local Translation Memory")
        st.markdown("- ✅ Sarvam AI API Integration")
        st.markdown("- ✅ Multi-language Support")
        st.markdown("- ✅ Real-time Translation")
        st.markdown("- ✅ Translation History")
    
    # Main translation interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔤 Input")
        
        # Language selection
        source_lang = st.selectbox(
            "Source Language",
            options=["en-IN", "hi-IN", "ta-IN"],
            format_func=lambda x: {
                "en-IN": "🇬🇧 English (India)",
                "hi-IN": "🇮🇳 Hindi (India)", 
                "ta-IN": "🇮🇳 Tamil (India)"
            }[x],
            key="source_lang"
        )
        
        # Text input
        input_text = st.text_area(
            "Enter text to translate:",
            height=150,
            placeholder="Type your text here...",
            key="input_text"
        )
        
        # Target language selection
        target_lang = st.selectbox(
            "Target Language",
            options=["en-IN", "hi-IN", "ta-IN"],
            format_func=lambda x: {
                "en-IN": "🇬🇧 English (India)",
                "hi-IN": "🇮🇳 Hindi (India)",
                "ta-IN": "🇮🇳 Tamil (India)"
            }[x],
            index=1 if source_lang == "en-IN" else 0,
            key="target_lang"
        )
        
        # Translation button
        translate_btn = st.button("🔄 Translate", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📝 Output")
        
        if translate_btn and input_text.strip():
            if source_lang == target_lang:
                st.warning("⚠️ Source and target languages are the same!")
            else:
                # Show translation progress
                with st.spinner("🔄 Translating..."):
                    start_time = time.time()
                    
                    try:
                        # Perform translation
                        translation = translator.translate(input_text.strip(), source_lang, target_lang)
                        
                        end_time = time.time()
                        translation_time = end_time - start_time
                        
                        # Display results
                        st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                        st.text_area(
                            "Translation:",
                            value=translation,
                            height=150,
                            disabled=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Translation metadata
                        col_time, col_method = st.columns(2)
                        with col_time:
                            st.metric("⏱️ Time", f"{translation_time:.2f}s")
                        with col_method:
                            # Determine translation method
                            if translation == input_text.strip():
                                method = "🔄 No Translation"
                            else:
                                # Check if it was from database or API
                                exact_match, _ = translator.search_exact_match(input_text.strip(), source_lang, target_lang)
                                if exact_match:
                                    method = "💾 Database"
                                else:
                                    method = "🌐 Sarvam AI"
                            st.metric("📡 Method", method)
                        
                        # Success message
                        if translation != input_text.strip():
                            st.markdown(
                                '<div class="success-box">✅ Translation completed successfully!</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="info-box">ℹ️ No translation available - returned original text.</div>',
                                unsafe_allow_html=True
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Translation failed: {str(e)}")
                        logger.error(f"Translation error: {str(e)}")
        
        elif translate_btn:
            st.warning("⚠️ Please enter some text to translate!")
    
    # Translation examples section
    st.markdown("---")
    st.subheader("💡 Try These Examples")
    
    examples = [
        {"text": "Hello", "from": "en-IN", "to": "hi-IN", "desc": "Simple greeting"},
        {"text": "Thank you", "from": "en-IN", "to": "ta-IN", "desc": "Polite expression"},
        {"text": "Good morning", "from": "en-IN", "to": "hi-IN", "desc": "Time-based greeting"},
        {"text": "How are you?", "from": "en-IN", "to": "ta-IN", "desc": "Question"},
        {"text": "मेल", "from": "hi-IN", "to": "en-IN", "desc": "Hindi word"},
        {"text": "ஆவணம்", "from": "ta-IN", "to": "en-IN", "desc": "Tamil word"},
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(
                f"'{example['text']}'\n{example['from']} → {example['to']}\n({example['desc']})",
                key=f"example_{i}",
                use_container_width=True
            ):
                # Set the example in session state
                st.session_state.source_lang = example['from']
                st.session_state.target_lang = example['to']
                st.session_state.input_text = example['text']
                st.rerun()
    
    # Add tabs for better organization
    st.markdown("---")
    
    # Create tabs for recent translations and full export
    tab1, tab2 = st.tabs(["📚 Recent Translations", "📊 Translation Memory Export"])
    
    with tab1:
        st.subheader("Recent Translations (Last 10)")
        
        try:
            # Get recent translations from database
            triplets = translator.get_all_triplets()
            
            if triplets:
                # Show last 10 translations
                recent_triplets = sorted(triplets, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
                
                # Create a DataFrame for display
                display_data = []
                for triplet in recent_triplets:
                    english = triplet.get('english', '') or ''
                    tamil = decode_unicode(triplet.get('tamil', '')) or ''
                    hindi = decode_unicode(triplet.get('hindi', '')) or ''
                    context = triplet.get('context', '') or ''
                    
                    display_data.append({
                        'ID': triplet.get('id', ''),
                        'English': english[:50] + ('...' if len(english) > 50 else ''),
                        'Tamil': tamil[:50] + ('...' if len(tamil) > 50 else ''),
                        'Hindi': hindi[:50] + ('...' if len(hindi) > 50 else ''),
                        'Context': context[:30] + ('...' if len(context) > 30 else ''),
                        'Timestamp': triplet.get('timestamp', '')
                    })
                
                df = pd.DataFrame(display_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Quick download for recent translations
                csv_recent = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Recent Translations (CSV)",
                    csv_recent,
                    "recent_translations.csv",
                    "text/csv",
                    key='download-recent'
                )
            else:
                st.info("No translations available yet.")
                
        except Exception as e:
            st.error(f"Error loading recent translations: {str(e)}")
    
    with tab2:
        st.subheader("Complete Translation Memory Export")
        st.markdown("Download all translation entries from the database as a comprehensive CSV file.")
        
        try:
            # Get ALL translations from database
            all_triplets = translator.get_all_triplets()
            
            if all_triplets:
                # Statistics
                total_entries = len(all_triplets)
                en_count = sum(1 for t in all_triplets if t.get('english'))
                ta_count = sum(1 for t in all_triplets if t.get('tamil'))
                hi_count = sum(1 for t in all_triplets if t.get('hindi'))
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Entries", total_entries)
                with col2:
                    st.metric("English Entries", en_count)
                with col3:
                    st.metric("Tamil Entries", ta_count)
                with col4:
                    st.metric("Hindi Entries", hi_count)
                
                # Create comprehensive DataFrame
                export_data = []
                for triplet in all_triplets:
                    english = triplet.get('english', '') or ''
                    tamil = decode_unicode(triplet.get('tamil', '')) or ''
                    hindi = decode_unicode(triplet.get('hindi', '')) or ''
                    context = triplet.get('context', '') or ''
                    timestamp = triplet.get('timestamp', '') or ''
                    
                    export_data.append({
                        'ID': triplet.get('id', ''),
                        'English': english,
                        'Tamil': tamil,
                        'Hindi': hindi,
                        'Context': context,
                        'Timestamp': timestamp,
                        'Has_English': bool(english),
                        'Has_Tamil': bool(tamil),
                        'Has_Hindi': bool(hindi),
                        'Languages_Count': sum([bool(english), bool(tamil), bool(hindi)]),
                        'Entry_Type': 'Complete_Triplet' if all([english, tamil, hindi]) else 'Partial_Entry'
                    })
                
                full_df = pd.DataFrame(export_data)
                
                # Show preview of the data
                st.markdown("### 📋 Data Preview (First 5 Rows)")
                preview_df = full_df.head().copy()
                # Truncate long text for preview
                for col in ['English', 'Tamil', 'Hindi', 'Context']:
                    preview_df[col] = preview_df[col].apply(lambda x: str(x)[:50] + ('...' if len(str(x)) > 50 else '') if x else '')
                
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                
                # Export options
                st.markdown("### 📤 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Full export
                    full_csv = full_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Complete Translation Memory (CSV)",
                        full_csv,
                        f"complete_translation_memory_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key='download-full',
                        help=f"Downloads all {total_entries} translation entries with detailed metadata"
                    )
                
                with col2:
                    # Complete triplets only
                    complete_triplets = full_df[full_df['Entry_Type'] == 'Complete_Triplet']
                    if len(complete_triplets) > 0:
                        complete_csv = complete_triplets.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Complete Triplets Only (CSV)",
                            complete_csv,
                            f"complete_triplets_only_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key='download-complete-only',
                            help=f"Downloads {len(complete_triplets)} entries that have translations in all three languages"
                        )
                    else:
                        st.info("No complete triplets (with all 3 languages) found in the database.")
                
                # Additional export formats
                st.markdown("### 📋 Language-Specific Exports")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # English entries
                    english_entries = full_df[full_df['Has_English'] == True]
                    if len(english_entries) > 0:
                        english_csv = english_entries.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            f"📥 English Entries ({len(english_entries)})",
                            english_csv,
                            f"english_entries_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key='download-english'
                        )
                
                with col2:
                    # Tamil entries
                    tamil_entries = full_df[full_df['Has_Tamil'] == True]
                    if len(tamil_entries) > 0:
                        tamil_csv = tamil_entries.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            f"📥 Tamil Entries ({len(tamil_entries)})",
                            tamil_csv,
                            f"tamil_entries_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key='download-tamil'
                        )
                
                with col3:
                    # Hindi entries
                    hindi_entries = full_df[full_df['Has_Hindi'] == True]
                    if len(hindi_entries) > 0:
                        hindi_csv = hindi_entries.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            f"📥 Hindi Entries ({len(hindi_entries)})",
                            hindi_csv,
                            f"hindi_entries_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key='download-hindi'
                        )
                
                # Export summary
                st.markdown("### 📊 Export Summary")
                st.info(f"""
                **Available Export Options:**
                - **Complete Database**: {total_entries} total entries
                - **Complete Triplets**: {len(complete_triplets)} entries with all 3 languages
                - **English Entries**: {en_count} entries with English text
                - **Tamil Entries**: {ta_count} entries with Tamil text  
                - **Hindi Entries**: {hi_count} entries with Hindi text
                
                **CSV Format**: All exports include ID, full text (no truncation), context, timestamp, and metadata columns.
                """)
                
            else:
                st.warning("No translation entries found in the database.")
                
        except Exception as e:
            st.error(f"Error loading translation memory: {str(e)}")
            logger.error(f"Export error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🌐 Advanced Translation System | Powered by Sarvam AI & Local Translation Memory</p>
            <p>Supports English (India), Hindi (India), and Tamil (India)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 