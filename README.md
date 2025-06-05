# Minimalist Translation System

A lightweight, efficient translation system with Translation Memory that supports English, Hindi, and Tamil.

## Architecture

This system implements a minimalist translation memory (TM) approach with intelligent fallback to API-based translation:

1. **Translation Memory Store (TM)**: Stores known phrase triplets in English, Tamil, and Hindi
2. **Lookup Layer**: Checks if input text exists in TM for fast retrieval
3. **In-Context Translation Layer**: Uses examples from TM to guide translation (placeholder for LLM implementation)
4. **Fallback Translation Layer**: Uses Sarvam API for translations not found in memory
5. **Memory Update**: Automatically stores new translations to expand the TM

```
         ┌────────────────────┐
         │ User inputs text   │
         └────────┬───────────┘
                  │
       ┌──────────▼────────────┐
       │ Check in Translation  │
       │ Memory (JSON/SQLite)  │
       └────────┬──────────────┘
         Match Found?   No
             │           │
           Yes           ▼
             │   ┌────────────────────────────┐
             ▼   │ Generate in-context prompt │
     ┌─────────┐ │ and use LLM to translate   │
     │ Return  │ └─────────────┬──────────────┘
     │ stored  │               │
     │ output  │       Model Success?   No
     └─────────┘            │            │
                           Yes           ▼
                            │    ┌─────────────────────┐
                            ▼    │ Use fallback MT API │
                      ┌─────────────┐  (e.g., Sarvam)  │
                      │ Translation │ └────────────────┘
                      │ completed   │
                      └─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ Store new triplet in TM   │
              └───────────────────────────┘
                            │
                            ▼
                ┌────────────────────┐
                │ Return translation │
                └────────────────────┘
```

## Implementations

The system includes two implementations:

1. **SQLite-based**: `translation_system.py` - Light, file-based storage suitable for local development
2. **PostgreSQL-based**: `pg_translation_system.py` - Scalable database storage for production use

## Features

- **Efficient Translation Memory**: Fast lookup and fuzzy matching for similar phrases
- **Language Detection**: Automatic detection of Hindi and Tamil scripts
- **Triplet Completion**: Automatically generate translations for all supported languages
- **Intelligent Fallback**: Uses Sarvam API when needed
- **Web Interface**: Streamlit-based UI for easy interaction

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Both the SQLite and PostgreSQL implementations include a CLI for direct use:

```bash
# SQLite version
python translation_system.py

# PostgreSQL version
python pg_translation_system.py
```

### Web Interface

```bash
streamlit run web_interface.py
```

## Configuration

- **SQLite**: No configuration required, uses local file `translation_memory.db`
- **PostgreSQL**: Configure connection details in `pg_translation_system.py`
- **Sarvam API**: API key is configured in both implementation files

## Future Improvements

- Implement the in-context translation layer using an LLM
- Add support for additional languages
- Enhance fuzzy matching with embeddings and retrieval (FAISS)
- Implement caching for improved performance 