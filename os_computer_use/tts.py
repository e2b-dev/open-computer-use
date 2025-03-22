import os
from elevenlabs import ElevenLabs, save


client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

def _generate_mp3_from_text(text):
    return client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

def save_text_as_audio_file(text, file_path):
    audio = _generate_mp3_from_text(text)
    save(audio, file_path)