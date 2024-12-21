import pyttsx3

def text_to_speech_pyttsx3(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error: {e}")

# Example usage
text = "Hello, I am your metahuman assistant."
text_to_speech_pyttsx3(text)
