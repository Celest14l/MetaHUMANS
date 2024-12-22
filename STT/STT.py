import pyttsx3
def give_output(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


import speech_recognition as sr
def take_input():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as an audio source
    with sr.Microphone() as source:
        print("Say something!")
        audio = recognizer.listen(source)

    # Save or process the audio
    try:
        print("You said: " + recognizer.recognize_google(audio))
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Error with request: {e}")
    output=recognizer.recognize_google(audio)

take_input()
