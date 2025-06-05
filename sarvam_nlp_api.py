import os
import requests
import logging
from typing import Dict, List, Union, Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sarvam API Configuration
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    logger.warning("SARVAM_API_KEY not found in environment variables. Please set it.")

class SarvamEndpoints:
    """Sarvam AI API endpoints"""
    TRANSLATE = "https://api.sarvam.ai/translate"
    TRANSLITERATE = "https://api.sarvam.ai/transliterate"
    LANGUAGE_ID = "https://api.sarvam.ai/text-lid"

class LanguageCode(Enum):
    """Supported language codes in Sarvam AI API"""
    ENGLISH = "en-IN"
    HINDI = "hi-IN"
    BENGALI = "bn-IN"
    GUJARATI = "gu-IN"
    KANNADA = "kn-IN"
    MALAYALAM = "ml-IN"
    MARATHI = "mr-IN"
    ODIA = "od-IN"
    PUNJABI = "pa-IN"
    TAMIL = "ta-IN"
    TELUGU = "te-IN"
    AUTO = "auto"  # For automatic language detection

class ScriptCode(Enum):
    """Supported script codes in Sarvam AI"""
    LATIN = "Latn"      # Latin (Romanized script)
    DEVANAGARI = "Deva"  # Devanagari (Hindi, Marathi)
    BENGALI = "Beng"    # Bengali
    GUJARATI = "Gujr"   # Gujarati
    KANNADA = "Knda"    # Kannada
    MALAYALAM = "Mlym"  # Malayalam
    ODIA = "Orya"       # Odia
    GURMUKHI = "Guru"   # Gurmukhi
    TAMIL = "Taml"      # Tamil
    TELUGU = "Telu"     # Telugu

class OutputScript(Enum):
    """Output script options for translation"""
    ROMAN = "roman"                  # Transliteration in Romanized script
    FULLY_NATIVE = "fully-native"    # Transliteration in native script with formal style
    SPOKEN_NATIVE = "spoken-form-in-native"  # Transliteration in native script with spoken style

class TranslationMode(Enum):
    """Translation mode options"""
    FORMAL = "formal"
    MODERN_COLLOQUIAL = "modern-colloquial"
    CLASSIC_COLLOQUIAL = "classic-colloquial"
    CODE_MIXED = "code-mixed"

class NumeralsFormat(Enum):
    """Numerals format options for translation"""
    INTERNATIONAL = "international"  # Regular numerals (0-9)
    NATIVE = "native"               # Language-specific native numerals

class SpeakerGender(Enum):
    """Speaker gender options for translation"""
    MALE = "Male"
    FEMALE = "Female"

class SarvamNLPClient:
    """Client for Sarvam AI NLP APIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Sarvam NLP client.
        
        Args:
            api_key: API subscription key for Sarvam AI. If None, looks for SARVAM_API_KEY in environment.
        """
        self.api_key = api_key or SARVAM_API_KEY
        if not self.api_key:
            raise ValueError("Sarvam API key is required. Provide it or set SARVAM_API_KEY in environment.")
        
        self.headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def translate(self, 
                 input_text: str,
                 source_language: Union[str, LanguageCode] = LanguageCode.AUTO,
                 target_language: Union[str, LanguageCode] = LanguageCode.ENGLISH,
                 mode: Optional[Union[str, TranslationMode]] = None,
                 output_script: Optional[Union[str, OutputScript]] = None,
                 numerals_format: Optional[Union[str, NumeralsFormat]] = None,
                 speaker_gender: Optional[Union[str, SpeakerGender]] = None,
                 enable_preprocessing: bool = False) -> Dict:
        """
        Translate text from source language to target language.
        
        Args:
            input_text: The text to translate (≤1000 characters)
            source_language: The source language code or auto for detection
            target_language: The target language code
            mode: Translation mode (formal, modern-colloquial, classic-colloquial, code-mixed)
            output_script: Output script for transliteration style
            numerals_format: Format for numerals (international or native)
            speaker_gender: Gender of speaker (for better translations in code-mixed mode)
            enable_preprocessing: Enable custom preprocessing for better translations
            
        Returns:
            Dict containing translated_text, source_language_code, and request_id
        """
        if isinstance(source_language, LanguageCode):
            source_language = source_language.value
        
        if isinstance(target_language, LanguageCode):
            target_language = target_language.value
            
        if isinstance(mode, TranslationMode):
            mode = mode.value
            
        if isinstance(output_script, OutputScript):
            output_script = output_script.value
            
        if isinstance(numerals_format, NumeralsFormat):
            numerals_format = numerals_format.value
            
        if isinstance(speaker_gender, SpeakerGender):
            speaker_gender = speaker_gender.value
        
        payload = {
            "input": input_text,
            "source_language_code": source_language,
            "target_language_code": target_language,
        }
        
        # Add optional parameters if provided
        if mode:
            payload["mode"] = mode
            
        if output_script:
            payload["output_script"] = output_script
            
        if numerals_format:
            payload["numerals_format"] = numerals_format
            
        if speaker_gender:
            payload["speaker_gender"] = speaker_gender
            
        if enable_preprocessing:
            payload["enable_preprocessing"] = enable_preprocessing
        
        try:
            response = requests.post(
                SarvamEndpoints.TRANSLATE,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Translation successful: {source_language} → {target_language}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Translation API error: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
    
    def transliterate(self,
                     input_text: str,
                     source_language: Union[str, LanguageCode] = LanguageCode.AUTO,
                     target_language: Union[str, LanguageCode] = LanguageCode.ENGLISH) -> Dict:
        """
        Transliterate text from one script to another while preserving pronunciation.
        
        Args:
            input_text: The text to transliterate
            source_language: The source language code or auto for detection
            target_language: The target language code
            
        Returns:
            Dict containing transliterated_text, source_language_code, and request_id
        """
        if isinstance(source_language, LanguageCode):
            source_language = source_language.value
        
        if isinstance(target_language, LanguageCode):
            target_language = target_language.value
        
        payload = {
            "input": input_text,
            "source_language_code": source_language,
            "target_language_code": target_language
        }
        
        try:
            response = requests.post(
                SarvamEndpoints.TRANSLITERATE,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Transliteration successful: {source_language} → {target_language}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Transliteration API error: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
    
    def identify_language(self, input_text: str) -> Dict:
        """
        Identify the language and script of the input text.
        
        Args:
            input_text: The text for language identification
            
        Returns:
            Dict containing language_code, script_code, and request_id
        """
        payload = {
            "input": input_text
        }
        
        try:
            response = requests.post(
                SarvamEndpoints.LANGUAGE_ID,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Language identification successful: {result.get('language_code', 'unknown')}, {result.get('script_code', 'unknown')}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Language identification API error: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise

def get_language_name(language_code: str) -> str:
    """Get the language name from its code"""
    language_names = {
        "en-IN": "English",
        "hi-IN": "Hindi",
        "bn-IN": "Bengali",
        "gu-IN": "Gujarati",
        "kn-IN": "Kannada",
        "ml-IN": "Malayalam",
        "mr-IN": "Marathi",
        "od-IN": "Odia",
        "pa-IN": "Punjabi",
        "ta-IN": "Tamil",
        "te-IN": "Telugu"
    }
    return language_names.get(language_code, "Unknown")

def get_script_name(script_code: str) -> str:
    """Get the script name from its code"""
    script_names = {
        "Latn": "Latin (Romanized)",
        "Deva": "Devanagari",
        "Beng": "Bengali",
        "Gujr": "Gujarati",
        "Knda": "Kannada",
        "Mlym": "Malayalam",
        "Orya": "Odia",
        "Guru": "Gurmukhi",
        "Taml": "Tamil",
        "Telu": "Telugu"
    }
    return script_names.get(script_code, "Unknown") 