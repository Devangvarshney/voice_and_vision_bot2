import logging
import speech_recognition as sr
from pydub import AudioSegment
import os
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def record_audio(file_path="patient_voice_test.mp3", timeout=20, phrase_time_limit=None):
    """
    Records audio from the microphone, saves it as a WAV file, and then converts it to an MP3 file.

    Args:
        file_path (str): Path to save the recorded MP3 file.
        timeout (int): Max time to wait for speech start (seconds).
        phrase_time_limit (int): Max duration for recorded phrase (seconds).
    """
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")

            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")

            # Save as WAV first
            wav_path = "temp_audio.wav"
            with open(wav_path, "wb") as f:
                f.write(audio_data.get_wav_data())

            # Convert WAV to MP3 using libmp3lame
            audio_segment = AudioSegment.from_wav(wav_path)
            audio_segment.export(file_path, format="mp3", parameters=["-acodec", "libmp3lame"])

            logging.info(f"✅ Audio saved to {file_path}")

            # Cleanup: Remove temporary WAV file
            os.remove(wav_path)
            return file_path

    except Exception as e:
        logging.error(f"❌ An error occurred: {e}")
        return None


def trascribe_with_groq(audio_filepath, stt_model="whisper-large-v3", GROQ_API_KEY=None, language=None):
    """Transcribe audio with Groq's Whisper model with language support"""
    client = Groq(api_key=GROQ_API_KEY)

    params = {
        "model": stt_model,
        "file": open(audio_filepath, "rb")
    }

    # Add language if specified (otherwise whisper will auto-detect)
    if language:
        params["language"] = language

    transcription = client.audio.transcriptions.create(**params)
    return transcription.text
