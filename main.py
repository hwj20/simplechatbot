import time
import re
import openai
import requests
import pyaudio
import wave
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import pandas as pd
from openai import OpenAI
import keyboard

# OpenAI API
openai.api_key = "Your key"

previous_answers = []
# Initialize Google Speech Client
client = speech.SpeechClient()

# Setup audio recording
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1


def record_audio():
    # Initialize PyAudio
    audio_interface = pyaudio.PyAudio()

    # Open audio stream
    stream = audio_interface.open(format=FORMAT, channels=CHANNELS,
                                  rate=RATE, input=True,
                                  frames_per_buffer=CHUNK)
    # Record until a key is pressed
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # Check if any key is pressed to stop recording
        if keyboard.is_pressed("space"):
            print("Recording stopped.")
            break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()

    # Save the recorded audio to a file
    audio_file = "temp_audio.wav"
    wf = wave.open(audio_file, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    return audio_file


def transcribe_audio(audio_file):
    # Transcribe audio using Google ASR
    with open(audio_file, "rb") as f:
        audio_content = f.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US"
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        return None
    return response.results[0].alternatives[0].transcript


def analyze_excel(file_path):
    # Analyze the Excel file
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Get a summary of the data
        summary = df.describe(include="all")
        print("Data Summary:\n", summary)

        # Get basic information about the data
        info = df.info()

        return summary, info
    except Exception as e:
        return f"Error analyzing file: {e}"


# Used to store previous conversations
previous_conversations = []


def generate_answer(prompt, file_content):
    # Save the current user input to the conversation context
    previous_conversations.append({"role": "user", "content": prompt})

    try:
        # Set the API key
        client = openai.OpenAI(
            api_key=openai.api_key
        )

        # Build the conversation context, including all previous messages and new system messages
        messages = [{"role": "system", "content": f"IMPORTANT: the maximum of answer words is 50"
                                                  f"You are assigned to discuss the following questions with user:"
                                                  f"Using the class data table {file_content}, "
                                                  f"what are the three most common concerns for GenAI at the level of industry and at the level of employer/company? "
                                                  f"Also using the class data table, what are the three most common applications of GenAI at the level of industry and at the level of employer/company?"
                                                  f"With your partner, compare and contrast the specific guidelines for your graduate programs, professions and target companies – What is the same? What is different? "
                                                  f"Your previous conversations are"
                     }] + previous_conversations

        # Call the OpenAI API using the GPT model to generate an answer
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100
        )

        # Get the generated answer and add it to the conversation context
        generated_answer = completion.choices[0].message.content.strip()
        previous_conversations.append({"role": "assistant", "content": generated_answer})

        # Output the generated answer and current conversation context
        print(f"Generated Answer: {generated_answer}")
        print(f"Conversation History: {previous_conversations}")

        return generated_answer
    except Exception as e:
        return f"API call error: {e}"


def text_to_speech(text):
    # Create TTS client
    tts_client = texttospeech.TextToSpeechClient()

    # Set TTS input content
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    # Output audio in LINEAR16 format (WAV format)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Call the TTS API
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save as WAV format
    with open("response_audio.wav", "wb") as out:
        out.write(response.audio_content)
        print("Audio content written to file 'response_audio.wav'")

    return "response_audio.wav"


import sounddevice as sd
import soundfile as sf


def play_audio(file):
    # Read the WAV file
    data, fs = sf.read(file, dtype='float32')
    # Play the audio
    sd.play(data, fs)
    sd.wait()  # Wait for the audio to finish playing


transcripts = []


def main():
    # file_content = sheets_to_string(read_all_sheets('Your file path'))
    file_content = ""
    print(file_content)
    while True:
        # Press any key to start recording
        input("Press any key to start recording...")
        print("Recording... Press space to stop recording.")

        # Start recording
        audio_file = record_audio()

        # Transcribe the audio
        transcript = transcribe_audio(audio_file)

        # Check if there is a valid transcription result
        if not transcript:
            print("No audio detected. Please try speaking again.")
            continue

        print(f"User said: {transcript}")
        # Save the transcription result
        transcripts.append(transcript)

        # Press any key to end the recording and transcription
        end_recording = input("Press 'G' to quit and generate answer.")

        # If the user presses 'g', exit recording and generate an answer
        if end_recording.lower() == 'g':
            # Concatenate all transcription content into a single string
            full_transcript = " ".join(transcripts)
            print('=' * 50)
            print(f"User Full Transcript: {full_transcript}")
            print('=' * 50)

            # Generate an answer using the full transcript
            answer = generate_answer(full_transcript, file_content)
            print(f"GPT Answer: {answer}")

            # Convert the answer to speech and play it
            tts_audio = text_to_speech(answer)
            if tts_audio:
                play_audio(tts_audio)
            else:
                print("TTS failed to generate audio.")

            # Clear the transcription content
            transcripts.clear()

        time.sleep(1)  # Add a small delay


if __name__ == "__main__":
    main()


def read_all_sheets(file_path):
    # Read all the sheets from the Excel file
    try:
        sheets = pd.read_excel(file_path,
                               sheet_name=None)  # Returns a dictionary with sheet names as keys and DataFrame as values
        return sheets
    except Exception as e:
        return f"Error reading file: {e}"


def clean_text(text):
    # Remove large spaces, but keep single spaces between words
    cleaned_text = re.sub(r'\s{2,}', ' ', text)  # Replace two or more spaces with a single space
    return cleaned_text


def sheets_to_string(sheets):
    # Convert each sheet data into a string, clean extra spaces, and remove NaN
    data_strings = {}
    for sheet_name, df in sheets.items():
        # Remove rows or columns containing NaN, you can choose to use fillna to replace
        df_cleaned = df.dropna(how='all')  # Drop rows where all elements are NaN
        df_cleaned = df_cleaned.fillna('')  # Fill NaN with empty string

        # Convert to string and clean extra spaces
        df_string = df_cleaned.to_string(index=False)  # Set index=False to remove the index column
        cleaned_string = clean_text(df_string)  # Clean extra spaces
        data_strings[sheet_name] = cleaned_string
    return data_strings
