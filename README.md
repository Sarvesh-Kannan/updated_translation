# Multi-Language Translator with Translation Memory

A Streamlit-based translation application that supports English, Hindi, and Tamil translations using the Sarvam AI API and a local translation memory system for efficient and cost-effective translations.

## Features

- Support for English (en-IN), Hindi (hi-IN), and Tamil (ta-IN)
- Local translation memory to reduce API calls
- Properties and TMX file support for bulk translations
- Automatic language detection
- Word-by-word and sentence-level translations
- Translation history view
- Statistics dashboard

## Requirements

```bash
streamlit
pandas
requests
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Sarvesh-Kannan/updated_translation.git
cd updated_translation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Select source and target languages
2. Enter text to translate
3. View translation history in the "View Translations" tab
4. Monitor translation statistics in the sidebar

## Translation Memory

The system uses a local SQLite database to store translations, which helps reduce API calls and improves response time. It supports:

- Loading translations from .properties files
- Loading translations from TMX files
- Automatic storage of API translations
- Word-by-word translation capability

## License

MIT License 