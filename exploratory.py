import google.generativeai as genai
import base64
import sounddevice as sd
import numpy as np
import wave
from dotenv import load_dotenv
from google.cloud import texttospeech
import os
import pyaudio
import json
import subprocess
import time
import logging
import traceback

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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

chat_history = []
chat_session = model.start_chat(history=chat_history)

# Audio Recording Settings
fs = 44100  # Sample rate
silence_threshold = 200  # Adjust as needed (lower = more sensitive)
silence_seconds = 3
max_inactivity_seconds = 10  # Timeout duration in seconds

# ------------------- Alert Sound --------------------
def play_alert_sound(alert_type="listening"):
    """Play different sounds for different alert types."""
    beep_duration = 0.5  # seconds
    if alert_type == "listening":
        print("🔴 Listening... Speak now!")
        beep_frequency = 1000  # Hz (Higher pitch for listening)
    elif alert_type == "analyzing":
        print("🔵 Analyzing... Please wait for response.")
        beep_frequency = 500  # Hz (Lower pitch for analyzing)
    elif alert_type == "sleeping":
        print("🛏️ Waiting... Please say 'knock knock' to continue.")
        beep_frequency = 1500  # Hz (Medium-high pitch for sleeping)
    else:
        beep_frequency = 1000  # Default to a common frequency

    # Generate sound wave (sine wave)
    t = np.linspace(0, beep_duration, int(fs * beep_duration), endpoint=False)  # Time array
    beep_signal = np.sin(2 * np.pi * beep_frequency * t)  # Sine wave

    # Normalize the signal to be within the range of -1 to 1
    beep_signal = beep_signal * 0.5  # Reduce the amplitude to prevent clipping

    # Play the sound
    sd.play(beep_signal, samplerate=fs)
    sd.wait()

def is_silent(data):
    print(np.abs(data).mean(), silence_threshold)
    return np.abs(data).mean() < silence_threshold

def record_with_silence_detection(max_record_time=2):
    play_alert_sound("sleeping")
    frames = []
    silence_counter = 0
    chunk_duration = 0.5
    max_chunks = int(max_record_time / chunk_duration)

    for _ in range(max_chunks):
        chunk = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()
        frames.append(chunk)
    audio_data = np.concatenate(frames, axis=0)
    return audio_data

def record_until_silence():
    play_alert_sound("listening")
    print("🟣 Recording visitor message (it will stop when silent for 3s)...")
    frames = []
    silence_counter = 0
    chunk_duration = 0.5

    while True:
        chunk = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()
        frames.append(chunk)

        if is_silent(chunk):
            silence_counter += chunk_duration
        else:
            silence_counter = 0

        if silence_counter >= silence_seconds:
            break

    audio_data = np.concatenate(frames, axis=0)
    return audio_data

def save_wav(audio_data, filename="recorded_audio.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())
    return filename

def check_for_trigger(filename):
    with open(filename, "rb") as audio_file:
        audio_content = audio_file.read()

    audio_base64 = base64.b64encode(audio_content).decode("utf-8")

    contents = [
        {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_base64,
                    }
                },
                {"text": "If the audio includes the phrase 'knock knock', answer 'yes', otherwise answer 'no'."}
            ]
        }
    ]

    response = model.generate_content(contents)
    print("Trigger check response:", response.text)
    if "yes" in response.text.lower():
        subprocess.run(f'espeak "Who is there?"', shell=True)
        return True
    else:
        return False

def process_audio(filename):
    play_alert_sound("analyzing")
    with open(filename, "rb") as audio_file:
        audio_content = audio_file.read()

    audio_base64 = base64.b64encode(audio_content).decode("utf-8")

    contents = [
        {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_base64,
                    }
                },
                {"text": system_instruction}
            ]
        }
    ]

    try:
        response = model.generate_content(contents)
        json_part = json.loads(response.text.replace("json\n", "").replace("```", ""))
        chat_history.append({"role": "user", "parts": json_part[0]})
        chat_history.append({"role": "model", "parts": json_part[1]})
    except json.JSONDecodeError as e:
        print(f"Error in processing audio: {e}")
        json_part = ["", ""]
    except Exception as e:
        print(f"Unexpected error: {e}")
        json_part = ["", ""]
        # Log errors with traceback for better debugging
        logging.error(traceback.format_exc())
    
    print("Visitor: ", json_part[0])
    print("AI     : ", json_part[1])
    text_to_speech(json_part[1])
    last_interaction_time = time.time()
    #subprocess.run(f'espeak "{json_part[1]}"', shell=True)

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

# ------------------- Main Loop ----------------------

last_interaction_time = time.time()
hello_flag = False

try:
    while True:
        # Phase 1 - Waiting for "hello exploratory"
        if not hello_flag:
            audio_data = record_with_silence_detection(max_record_time=5)
            wav_file = save_wav(audio_data, filename="trigger_check.wav")
            hello_flag = check_for_trigger("trigger_check.wav")
        # Check if there is a valid trigger
        if hello_flag:
            # Phase 2 - Real recording
            print("✅ Trigger detected! Start recording visitor message.\n")
            visitor_audio = record_until_silence()
            visitor_wav = save_wav(visitor_audio, filename="visitor_message.wav")
            process_audio(visitor_wav)
            last_interaction_time = time.time()
            print(time.time() - last_interaction_time, max_inactivity_seconds)
            if time.time() - last_interaction_time > max_inactivity_seconds:
                print("No interaction for 10 seconds, clearing session.\n")
                hello_flag=False
                chat_history.clear()
                last_interaction_time = time.time()  # Reset the timer for the next cycle

except KeyboardInterrupt:
    print("\nShutting down gracefully...")
    # Cleanup code here if needed (e.g., saving states, releasing resources)
