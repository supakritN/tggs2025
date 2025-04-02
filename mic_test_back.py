import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import queue
import sounddevice as sd
import sys
import time
from vosk import Model, KaldiRecognizer
import google.generativeai as genai
import espeak
print(dir(espeak))
print(espeak.__file__)

GEMINI_API_KEY = "AIzaSyBtALzxNSqy6z5MuTQQ-i_jFBiDHH4LRfM"

genai.configure(api_key=GEMINI_API_KEY)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="You are a front-door AI assistant. Identify their reason for visiting and provide relevant information clearly.\n"
                    "- If they ask for a specific person, if they ask for \"Anek\",\"อเนก\" Notify the owner else tell them it's the wrong house.\n"
                    "- If it’s a delivery, instruct to leave the package at the table.\n"
                    "- If they ark for contcact infomation, the phone number is 077-7777777"
                    "Keep responses natural, concise, and polite. Respond in a calm, welcoming voice.",
)

chat_history = []
chat_session = model.start_chat(history=[])

def listen_and_transcribe(model, samplerate=16000, silence_timeout=2, chunk_duration=3):
    q = queue.Queue()
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)
    results = []
    last_speech_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    # Function to process audio chunks from the queue
    def process_audio():
        nonlocal last_speech_time
        while True:
            # Wait for the next chunk from the queue
            data = q.get()

            # Process the chunk of audio
            if rec.AcceptWaveform(data):
                res = rec.Result()
                if '"text"' in res:
                    text = eval(res)["text"]
                    if text.strip():
                        results.append(text)
                        last_speech_time = time.time()
                        print("add text")
            else:
                partial = eval(rec.PartialResult())["partial"]
                if partial.strip():
                    last_speech_time = time.time()

            # Check for silence timeout and stop if no speech is detected
            if time.time() - last_speech_time > silence_timeout:
                break

    # Start the audio input stream
    with sd.RawInputStream(samplerate=samplerate, blocksize=samplerate * chunk_duration, dtype='int16',
                           channels=1, callback=audio_callback):
        print("Listening...")

        # Keep capturing and processing audio chunks sequentially
        while True:
            # Pause recording and process the chunk
            process_audio()

            # Stop if the silence timeout condition is met
            if time.time() - last_speech_time > silence_timeout:
                break

    return " ".join(results)

def chat(user_input):
    response = chat_session.send_message(user_input)

    # Save conversation history
    chat_history.append({"role": "user", "parts": [user_input]})
    chat_history.append({"role": "model", "parts": [response.text]})

    print("AI:", response.text)
    os.popen("espeak "+ '/"'+response.text+'/"')
        
def setup():
    MODEL_PATH = "vosk-model-small-en-us-0.15"
    #MODEL_PATH = "vosk-model-en-us-0.22-lgraph"
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Download it and put it in the current directory.")
        return

    model_stt = Model(MODEL_PATH)
    print("Done setting...")
    return model_stt

def main():
    model_stt = setup()
    try:
        while True:
            transcript = listen_and_transcribe(model_stt)
            flag = True
            if transcript.strip():
                print("You said:", transcript)
                if flag:
                    chat(transcript)
                    flag=False
    except KeyboardInterrupt:
        print("Program interrupted")

if __name__ == "__main__":
    main()
