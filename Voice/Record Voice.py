import pyaudio
import wave
import keyboard

# Set parameters for audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream to capture audio from a tab
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Wait for the user to start the recording
print("Press 'r' to start recording...")
keyboard.wait("r")

# Record audio from the tab
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Stop recording
print("Recording stopped.")
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a file
wf = wave.open("recorded_audio.wav", "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()