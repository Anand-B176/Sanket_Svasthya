"""
Text-to-Speech Module - Sanket-Svasthya
Audio feedback for sign language recognition.

Author: Team Sanket-Svasthya
Date: January 2026
"""

import pyttsx3
import threading


class TTSEngine:
    """
    Thread-safe Text-to-Speech engine.
    
    Provides non-blocking speech synthesis for
    real-time sign language feedback.
    """
    
    def __init__(self, rate: int = 150, volume: float = 0.9):
        """
        Initialize TTS engine.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.rate = rate
        self.volume = volume
        self._lock = threading.Lock()
    
    def _create_engine(self) -> pyttsx3.Engine:
        """Create a new engine instance."""
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        return engine
    
    def speak(self, text: str, blocking: bool = False):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        if blocking:
            self._speak_sync(text)
        else:
            self._speak_async(text)
    
    def _speak_sync(self, text: str):
        """Synchronous speech (blocks until complete)."""
        with self._lock:
            try:
                engine = self._create_engine()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
    
    def _speak_async(self, text: str):
        """Asynchronous speech (non-blocking)."""
        thread = threading.Thread(
            target=self._speak_sync,
            args=(text,),
            daemon=True
        )
        thread.start()
    
    def set_voice(self, voice_id: int = 0):
        """
        Set voice by index.
        
        Args:
            voice_id: Voice index (0 = default male, 1 = female usually)
        """
        try:
            engine = self._create_engine()
            voices = engine.getProperty('voices')
            if voice_id < len(voices):
                engine.setProperty('voice', voices[voice_id].id)
        except Exception as e:
            print(f"Voice change error: {e}")
    
    def list_voices(self) -> list:
        """List available voices."""
        try:
            engine = self._create_engine()
            voices = engine.getProperty('voices')
            return [{'id': i, 'name': v.name} for i, v in enumerate(voices)]
        except:
            return []


# Convenience function
def speak(text: str):
    """Quick non-blocking speech."""
    tts = TTSEngine()
    tts.speak(text, blocking=False)


if __name__ == "__main__":
    # Test TTS
    print("Testing TTS engine...")
    
    tts = TTSEngine()
    
    print("Available voices:")
    for voice in tts.list_voices():
        print(f"  {voice['id']}: {voice['name']}")
    
    print("\nSpeaking test phrases...")
    tts.speak("Hello, this is Sanket Svasthya")
    tts.speak("Testing medical sign recognition", blocking=True)
    
    print("TTS test complete!")
