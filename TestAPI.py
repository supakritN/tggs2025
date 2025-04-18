import google.generativeai as genai
import base64
import sounddevice as sd
import numpy as np
import wave
from dotenv import load_dotenv
import os
import json
import re
import time

from google.cloud import texttospeech
import os
import pyaudio

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the API with the key
genai.configure(api_key=GEMINI_API_KEY)

# Create the model
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

system_instruction ="""
  "For the progrom I need you to create array as a responce in this one array [**'Base64 audio input transcript to text of What is said in the audio'**, **'Responce of upcoming instruction'**]
You are a front-door AI assistant. Your task is to identify the visitor's reason for coming 
and provide relevant information.
- If they ask for a specific person using the keywords **'Steve'** or **'Steven'**, notify the owner. 
  Otherwise, inform them that it’s the wrong house.
- If it’s a delivery and the keywords like **'Grab'**, **'Lineman'**, or **'food'** are mentioned, instruct them to leave 
  the package on the table.
- If they ask for contact information using the keywords like **'call'** or **'phone'**, provide the phone number: **077-7777777**.

Keep responses natural, concise, and polite. Respond in a calm and welcoming voice.
**Respond in Thai language.**
        """


print("System Instruction:")
print(system_instruction)
# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# Maintain chat history

chat_history = []
chat_session = model.start_chat(history=chat_history)

# Audio Recording Settings
fs = 44100  # Sample rate
silence_threshold = 10000  # Adjust as needed (lower = more sensitive)
max_record_seconds = 5  # Prevent recording forever
silence_duration = 2.0  # Seconds of silence before stopping

def record_audio():
    """Records audio until the user stops talking."""
    print("\n🎤 Recording... Speak now!")

    recording = []
    start_time = time.time()
    silence_start = None

    # Create a mutable holder for stream so it's accessible inside the callback
    stream_holder = {"stream": None}

    def callback(indata, frames, time_info, status):
        nonlocal recording, silence_start

        volume = np.linalg.norm(indata)
        print(f"Volume: {volume:.2f}")
        recording.append(indata.copy())

        if volume < silence_threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > silence_duration:
                print('🔇 Silence detected. Stopping recording...')
                stream_holder["stream"].stop()
        else:
            silence_start = None

    with sd.InputStream(callback=callback, channels=1, samplerate=fs, dtype=np.int16) as stream:
        stream_holder["stream"] = stream
        while stream.active and time.time() - start_time < max_record_seconds:
            sd.sleep(100)

    print("✅ Recording complete.")

    audio_data = np.concatenate(recording, axis=0)
    wav_filename = "recorded_audio.wav"

    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

    return wav_filename

def text_to_speech(text):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "iot-exploratory-2025-64d61984f617.json"
    
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="th-TH",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream using standard PCM 16-bit mono at 24000 Hz (Google LINEAR16 default)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True)

    # Play the raw audio
    stream.write(response.audio_content)

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()


def process_audio(filename):
    """Reads and encodes audio, then sends it to the AI model for processing."""
    with open(filename, "rb") as audio_file:
        audio_content = audio_file.read()

    audio_base64 = base64.b64encode(audio_content).decode("utf-8")

    # Create input content
    contents = [
        {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "audio/mpeg",  # Correct MIME type for MP3
                        "data": audio_base64,
                    }
                },
                {
                    "text": system_instruction  # Optional text prompt
                }
            ]
        }
    ]

    # Generate response
    response = model.generate_content(contents)

    # Save conversation to history
    if isinstance(response.text, str):
        json_part =  json.loads(response.text.replace("```json\n", "").replace("```", ""))
    try:
        chat_history.append({"role": "user", "parts": json_part[0]})
        chat_history.append({"role": "model", "parts": json_part[1]})
        model_response = json_part[1]
    except (SyntaxError, ValueError):
        print(SyntaxError, ValueError)
        model_response = "Try is broken"

    print("Visitor:", json_part[0])
    print("AI:", model_response)
    text_to_speech(model_response)
    
# Loop to wait for the 'R' key press
#print("\n🔹 Press 'R' to record audio. Press 'ESC' to exit.")
while True:
        audio_file = record_audio()
        process_audio(audio_file)
