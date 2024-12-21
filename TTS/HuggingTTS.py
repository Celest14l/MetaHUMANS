import speech_recognition as sr
from TTS.api import TTS
import time

# Initialize Coqui TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

# Initialize the Speech Recognition recognizer
recognizer = sr.Recognizer()


# Function to recognize speech using SpeechRecognition
def recognize_speech():
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Listen to the source (microphone)
        print("Recognizing...")

        try:
            # Using Google's speech recognition API
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            print("Could not request results; check your internet connection.")
            return None


# Function to convert text to speech and play it using Coqui TTS
def text_to_speech(text):
    print(f"AI says: {text}")
    tts.tts_to_file(text, "output.wav")  # Save the audio to a file
    # Play the audio file using an appropriate method for your OS
    # On Windows, we can use the default media player
    import os
    os.system("start output.wav")  # For Windows; use "open" on macOS


# Main AI function
def ai_conversation():
    while True:
        user_input = recognize_speech()  # Get user speech input

        if user_input is None:
            continue

        # Stop the loop if user says 'exit'
        if 'exit' in user_input.lower():
            print("Goodbye!")
            break

        # Basic AI response logic (can be expanded with any model, etc.)
        if 'how are you' in user_input.lower():
            response = "I'm doing great, thank you for asking!"
        elif 'your name' in user_input.lower():
            response = "I am your personal assistant!"
        else:
            response = "Sorry, I didn't understand that. Can you say it again?"

        text_to_speech(response)  # Convert the response to speech and play it


if __name__ == "__main__":
    ai_conversation()
