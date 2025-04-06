import subprocess
import platform
import os
from gtts import gTTS


def text_to_speech_with_gtts(input_text, output_filepath, language="en"):
    """Convert text to speech with language support for Hindi, English, Punjabi, and Malayalam"""
    # Map language codes to gTTS format
    language_map = {
        "en": "en",
        "hi": "hi",
        "pa": "pa",
        "ml": "ml",
        "auto": "en"  # Default to English for auto
    }

    # Use mapped language or default to English
    tts_language = language_map.get(language, "en")

    audioobj = gTTS(
        text=input_text,
        lang=tts_language,
        slow=False
    )
    audioobj.save(output_filepath)

    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")

    return output_filepath
