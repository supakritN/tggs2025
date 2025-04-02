import google.generativeai as genai
import base64
import sounddevice as sd
import numpy as np
import wave
from dotenv import load_dotenv
import os
import json
import re
import subprocess

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

system_instruction = (
    "For the progrom I need you to create array as a responce in this one array [**'Base64 audio input transcript to text of What is said in the audio'**, **'Responce of upcoming instruction'**]"
    "- If you don't understand the messege answer empty message."
    "You are a front-door AI assistant. Your task is to identify the visitor's reason for coming "
    "and provide relevant information. If recieve Thai messege, answer in English.\n"
    "- If they ask for a specific person using the keywords **'Steve'** or **'Steven'**, notify the owner. "
    "Otherwise, inform them that it’s the wrong house.\n"
    "- If it’s a delivery and the keywords like **'Grab'** ,**'Lineman'**, or **'food'** are mentioned, instruct them to leave "
    "the package on the table.\n"
    "- If they ask for contact information using the keywords like **'call'** or **'phone'**, provide the phone number: **077-7777777**.\n\n"
    "Keep responses natural, concise, and polite. Respond in a calm and welcoming voice."
)

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

# Maintain chat history
chat_history = []
chat_session = model.start_chat(history=chat_history)

# Audio Recording Settings
fs = 44100  # Sample rate
seconds = 5  # Duration of recording

def record_audio():
    """Records audio for a fixed duration and saves it as a WAV file."""
    print("\n🎤 Recording... Speak now!")
    recorded_audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    print("✅ Recording complete.")

    # Save as WAV file
    wav_filename = "recorded_audio.wav"
    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(fs)
        wf.writeframes(recorded_audio.tobytes())

    return wav_filename

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
    #os.popen(f"espeak \"{model_response}\"")
    subprocess.run(f"espeak \"{model_response}\"", shell=True)

# Loop to wait for the 'R' key press
#print("\n🔹 Press 'R' to record audio. Press 'ESC' to exit.")
while True:
        audio_file = record_audio()
        process_audio(audio_file)
