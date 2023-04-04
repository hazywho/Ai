import speech_recognition as sr
import os

# Path of the audio file to transcribe
audio_path = "recorded_audio.wav"

# Create a recognizer object
r = sr.Recognizer()

# Load the audio file
with sr.AudioFile(audio_path) as source:
    # Read the audio data from the file
    audio_data = r.record(source)
    
    # Perform speech recognition
    text = r.recognize_google(audio_data)

# Print the transcribed text
print(text)