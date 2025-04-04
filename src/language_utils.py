# src/language_utils.py
from langdetect import detect, DetectorFactory
from typing import Optional, List
import logging

# For more stable language detection
DetectorFactory.seed = 0

# Logger setup
logger = logging.getLogger(__name__)

class LanguageUtils:
    """Utility class for language operations"""
    
    SUPPORTED_LANGUAGES = {
        # Common European languages
        'en': 'English',
        'ru': 'Russian',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        
        # Asian languages
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        
        # Other widely used languages
        'ar': 'Arabic',
        'hi': 'Hindi',
        'tr': 'Turkish',
        'cs': 'Czech',
        'uk': 'Ukrainian',
        'bg': 'Bulgarian',
        'el': 'Greek',
        'he': 'Hebrew',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'hu': 'Hungarian',
        'sk': 'Slovak',
        'ro': 'Romanian',
        'id': 'Indonesian',
        'ms': 'Malay',
    }
    
    @classmethod
    def get_language_name(cls, lang_code: str) -> str:
        """Get language name from code"""
        return cls.SUPPORTED_LANGUAGES.get(lang_code, "Unknown")
    
    @classmethod
    def is_supported(cls, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in cls.SUPPORTED_LANGUAGES
    
    @classmethod
    def get_closest_supported_language(cls, lang_code: str) -> str:
        """
        Get the closest supported language code
        
        This helps with similar language detection issues
        like confusing 'no' (Norwegian) with 'da' (Danish)
        """
        if lang_code in cls.SUPPORTED_LANGUAGES:
            return lang_code
            
        # Language mapping for commonly confused languages
        similar_languages = {
            'nb': 'no',  # Norwegian Bokmål → Norwegian
            'nn': 'no',  # Norwegian Nynorsk → Norwegian
            'zh-cn': 'zh',  # Chinese Simplified → Chinese 
            'zh-tw': 'zh',  # Chinese Traditional → Chinese
            'hr': 'sr',  # Croatian → Serbian (similar)
            'bs': 'sr',  # Bosnian → Serbian (similar)
            'mk': 'bg',  # Macedonian → Bulgarian (similar)
            'be': 'ru',  # Belarusian → Russian (similar)
            'ca': 'es',  # Catalan → Spanish (similar)
            'gl': 'pt',  # Galician → Portuguese (similar)
            'af': 'nl',  # Afrikaans → Dutch (similar)
        }
        
        return similar_languages.get(lang_code, "en")
    
# Create instance for convenient import
language_processor = LanguageUtils()
