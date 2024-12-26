import google.generativeai as genai
import pyttsx3
import speech_recognition as sr

# Configure your Gemini API key
genai.configure(api_key='GEMINI_API_KEY')  # Replace with your actual API key

# Function to convert text to speech using pyttsx3
def give_output(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust speed of speech
    engine.setProperty('volume', 1)  # Adjust volume (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

# Function to get speech input using SpeechRecognition
def take_input(recognizer, source):
    print("Listening...")
    try:
        audio = recognizer.listen(source, timeout=5)  # Timeout for faster responses
        input_text = recognizer.recognize_google(audio)
        print(f"You said: {input_text}")
        return input_text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Error with request: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to continuously handle speech input and output
def speech_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Calibrate microphone to ambient noise
        print("Say something... (Say 'exit' or 'goodbye' to stop)")

        while True:
            input_text = take_input(recognizer, source)
            if input_text:
                if "exit" in input_text or "goodbye" in input_text:
                    print("Exiting...")
                    give_output("Goodbye! Have a great day!")
                    break

                # Get response from Google Generative AI
                try:
                    response = genai.chat(prompt=input_text)
                    assistant_response = response['content']  # Extract the generated content
                    print(f"Assistant: {assistant_response}")

                    # Speak the response
                    give_output(assistant_response)
                except Exception as e:
                    print(f"Error generating response: {e}")
                    give_output("Sorry, I couldn't process that.")

# Run the speech-to-speech communication
speech_to_speech()
