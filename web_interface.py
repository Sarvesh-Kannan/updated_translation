import streamlit as st
import pandas as pd
from translation_system import MinimalistTranslationSystem

def main():
    st.set_page_config(
        page_title="Minimalist Translation System",
        page_icon="🌏",
        layout="wide"
    )
    
    st.title("🌏 Minimalist Translation System")
    
    # Initialize the translation system
    if 'translator' not in st.session_state:
        st.session_state.translator = MinimalistTranslationSystem()
        st.session_state.history = []
    
    # Sidebar with system stats
    st.sidebar.title("System Information")
    
    # Add about information
    st.sidebar.markdown("""
    ### About
    
    This is a minimalist translation system that supports English, Hindi, and Tamil translations.
    
    #### Features:
    - Translation Memory (TM) for fast lookups
    - Automatic triplet completion
    - Fallback to Sarvam API
    - Fuzzy matching for similar texts
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Translate", "View Translations", "Complete Triplets"])
    
    # Tab 1: Translation Interface
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "Source Language",
                options=list(st.session_state.translator.supported_languages.keys()),
                format_func=lambda x: st.session_state.translator.supported_languages[x]
            )
            
            source_text = st.text_area(
                "Source Text",
                height=150,
                placeholder="Enter text to translate..."
            )
        
        with col2:
            target_options = [lang for lang in st.session_state.translator.supported_languages.keys() 
                             if lang != source_lang]
            
            target_lang = st.selectbox(
                "Target Language",
                options=target_options,
                format_func=lambda x: st.session_state.translator.supported_languages[x]
            )
            
            translate_button = st.button("Translate", type="primary")
            
            if translate_button and source_text:
                with st.spinner("Translating..."):
                    # Auto-detect language from script
                    detected_lang = None
                    if any('\u0B80' <= c <= '\u0BFF' for c in source_text):  # Tamil script
                        detected_lang = 'ta-IN'
                        st.info(f"Detected Tamil script. Source language set to Tamil.")
                    elif any('\u0900' <= c <= '\u097F' for c in source_text):  # Devanagari script
                        detected_lang = 'hi-IN'
                        st.info(f"Detected Hindi script. Source language set to Hindi.")
                    
                    used_source_lang = detected_lang or source_lang
                    
                    translation = st.session_state.translator.translate(
                        source_text, used_source_lang, target_lang
                    )
                    
                    # Add to history
                    st.session_state.history.append({
                        'source_lang': used_source_lang,
                        'source_text': source_text,
                        'target_lang': target_lang,
                        'target_text': translation,
                        'timestamp': pd.Timestamp.now()
                    })
            
            if st.session_state.history:
                latest = st.session_state.history[-1]
                st.text_area(
                    f"Translation ({st.session_state.translator.supported_languages[latest['target_lang']]})",
                    value=latest['target_text'],
                    height=150
                )
    
    # Tab 2: View Translations
    with tab2:
        st.subheader("Translation Memory")
        
        triplets = st.session_state.translator.get_all_triplets()
        
        if not triplets:
            st.info("No translations stored in the Translation Memory yet.")
        else:
            # Convert to DataFrame for display
            df = pd.DataFrame(triplets)
            df = df[['id', 'english', 'tamil', 'hindi', 'timestamp']]
            df.columns = ['ID', 'English', 'Tamil', 'Hindi', 'Timestamp']
            
            # Make timestamp more readable
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download as CSV",
                csv,
                "translation_memory.csv",
                "text/csv",
                key='download-csv'
            )
    
    # Tab 3: Complete Triplets
    with tab3:
        st.subheader("Complete Translation Triplets")
        st.markdown("""
        This feature allows you to enter text in one language and automatically translate it to all supported languages.
        The complete triplet will be stored in the Translation Memory.
        """)
        
        known_lang = st.selectbox(
            "Text Language",
            options=list(st.session_state.translator.supported_languages.keys()),
            format_func=lambda x: st.session_state.translator.supported_languages[x]
        )
        
        text_to_complete = st.text_area(
            "Enter Text",
            height=100,
            placeholder=f"Enter text in {st.session_state.translator.supported_languages[known_lang]}..."
        )
        
        if st.button("Complete Triplet", type="primary") and text_to_complete:
            with st.spinner("Completing triplet..."):
                triplet = st.session_state.translator.complete_triplet(text_to_complete, known_lang)
                
                st.success("Triplet completed successfully!")
                
                # Display the completed triplet
                st.markdown("### Completed Triplet")
                for lang, field in [('en-IN', 'english'), ('ta-IN', 'tamil'), ('hi-IN', 'hindi')]:
                    st.markdown(f"**{st.session_state.translator.supported_languages[lang]}**: {triplet[field]}")

if __name__ == "__main__":
    main() 