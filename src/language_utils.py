# src/language_utils.py
from langdetect import detect, DetectorFactory
from typing import Optional, List
import logging

# For more stable language detection
DetectorFactory.seed = 0

# Logger setup
logger = logging.getLogger(__name__)

class LanguageUtils:
    """Centralized class for language processing"""
    
    # Supported languages (can be extended)
    SUPPORTED_LANGUAGES = ["en", "ru", "uk", "de", "fr", "es", "it", "pt"]
    
    @classmethod
    def detect_language(cls, text: str, default: str = "en") -> str:
        """
        Detects text language with enhanced error handling
        
        Args:
            text: Text to analyze
            default: Default language in case of error
            
        Returns:
            Language code (ISO 639-1)
        """
        try:
            # Minimum length for reliable detection
            if len(text.strip()) < 15:
                logger.warning(f"Text too short for reliable detection: '{text}'")
                return default
                
            lang = detect(text)
            
            # Check language support
            if lang not in cls.SUPPORTED_LANGUAGES:
                logger.warning(f"Unsupported language detected: {lang}. Defaulting to {default}")
                return default
                
            logger.debug(f"Detected language: {lang} for text: '{text[:50]}...'")
            return lang
            
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}. Text: '{text[:100]}...'")
            return default
    
    @classmethod
    def get_language_instruction(cls, target_lang: str, user_message: str) -> str:
        """
        Generates strict response language instructions
        
        Args:
            target_lang: Language the bot should respond in
            user_message: Original user message
            
        Returns:
            String with prompt instructions
        """
        instructions = {
            "en": f"CRITICAL: Respond in English only. Never switch languages.\n\nOriginal message: {user_message}",
            "ru": f"ВАЖНО: Отвечайте только на русском. Не переключайтесь на другие языки.\n\nОригинальное сообщение: {user_message}",
            "uk": f"ВАЖЛИВО: Відповідайте лише українською. Не змінюйте мову.\n\nОригінальне повідомлення: {user_message}",
            "de": f"KRITISCH: Antworten Sie nur auf Deutsch. Wechseln Sie nie die Sprache.\n\nOriginalnachricht: {user_message}",
            "fr": f"CRITIQUE: Répondez uniquement en français. Ne changez jamais de langue.\n\nMessage original: {user_message}",
            "es": f"CRÍTICO: Responda sólo en español. Nunca cambie de idioma.\n\nMensaje original: {user_message}",
            "it": f"IMPORTANTE: Rispondere solo in italiano. Non cambiare lingua.\n\nMessaggio originale: {user_message}",
            "pt": f"CRÍTICO: Responda apenas em português. Nunca mude de idioma.\n\nMensagem original: {user_message}"
        }
        
        return instructions.get(target_lang, instructions["en"])
    
    @classmethod
    def validate_response_language(cls, response: str, expected_lang: str) -> bool:
        """
        Validates if response language matches expected language
        
        Args:
            response: Bot's response
            expected_lang: Expected language (ISO 639-1)
            
        Returns:
            True if language matches, False if not
        """
        try:
            detected_lang = cls.detect_language(response)
            if detected_lang != expected_lang:
                logger.warning(f"Language mismatch! Expected {expected_lang}, got {detected_lang}")
                return False
            return True
        except Exception as e:
            logger.error(f"Language validation failed: {str(e)}")
            return False

# Create instance for convenient import
language_processor = LanguageUtils()
