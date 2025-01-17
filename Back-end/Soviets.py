import os
import requests
import pyttsx3  # For text-to-speech and generating .wav files
import speech_recognition as sr  # For speech recognition
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import numpy as np
import random
from gradio_client import Client, handle_file
import shutil
# Set the Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Directory to store .wav files
AUDIO_DIR = r"D:\Projects\MetaHUMANS\responses_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)  # Create directory if it doesn't exist

# NVIDIA Audio2Face API base URL
A2F_BASE_URL = "http://localhost:8011"  # Replace with your server's URL

# Initialize VADER for text sentiment analysis
vader_analyzer = SentimentIntensityAnalyzer()


def analyze_text_sentiment(text):
    sentiment_score = vader_analyzer.polarity_scores(text)
    compound = sentiment_score['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def analyze_audio_sentiment(audio_path):
    try:
        [fs, x] = audioBasicIO.read_audio_file(audio_path)
        x = audioBasicIO.stereo_to_mono(x)  # Convert to mono
        features, _ = ShortTermFeatures.feature_extraction(x, fs, 0.05 * fs, 0.025 * fs)
        energy = np.mean(features[1])  # Extract energy as a basic feature
        if energy > 0.01:
            return "Positive"
        elif energy < 0.005:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print(f"Error analyzing audio sentiment: {e}")
        return "Neutral"


def randomize_emotion(base, variation=0.1):
    return round(random.uniform(max(0.5, base - variation), 1), 2)

def generate_emotion_payload(user_sentiment, chatbot_sentiment):
    emotions = {
        "amazement": 0,
        "anger": 0,
        "cheekiness": 0,
        "disgust": 0,
        "fear": 0,
        "grief": 0,
        "joy": 0,
        "outofbreath": 0,
        "pain": 0,
        "sadness": 0,
    }

    if user_sentiment == "Positive":
        if chatbot_sentiment == "Positive":
            emotions["joy"] = randomize_emotion(0.9, 0.2)
            emotions["amazement"] = randomize_emotion(0.6, 0.2)
        elif chatbot_sentiment == "Neutral":
            emotions["joy"] = randomize_emotion(0.7, 0.2)
        else:
            emotions["joy"] = randomize_emotion(0.5, 0.2)
            emotions["sadness"] = randomize_emotion(0.4, 0.2)

    elif user_sentiment == "Neutral":
        if chatbot_sentiment == "Positive":
            emotions["cheekiness"] = randomize_emotion(0.5, 0.2)
            emotions["joy"] = randomize_emotion(0.4, 0.1)
        elif chatbot_sentiment == "Neutral":
            emotions["cheekiness"] = randomize_emotion(0.3, 0.1)
        else:
            emotions["grief"] = randomize_emotion(0.4, 0.2)
            emotions["sadness"] = randomize_emotion(0.5, 0.2)

    elif user_sentiment == "Negative":
        if chatbot_sentiment == "Positive":
            emotions["sadness"] = randomize_emotion(0.4, 0.2)
            emotions["joy"] = randomize_emotion(0.4, 0.2)
        elif chatbot_sentiment == "Neutral":
            emotions["grief"] = randomize_emotion(0.5, 0.2)
        else:
            emotions["sadness"] = randomize_emotion(0.7, 0.2)
            emotions["grief"] = randomize_emotion(0.5, 0.2)

    return {
        "a2f_instance": "/World/audio2face/CoreFullface",
        "emotions": emotions,
    }
def check_api_status():
    try:
        response = requests.get(f"{A2F_BASE_URL}/status")
        response.raise_for_status()
        print("API Status:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error checking API status: {e}")


def load_usd_file():
    payload = {"file_name": r"D:\Projects\MetaHUMANS\priyanshu.usd"}
    try:
        response = requests.post(f"{A2F_BASE_URL}/A2F/USD/Load", json=payload)
        response.raise_for_status()
        result = response.json()
        print("USD file loaded successfully:", result)
        return result  # Ensure meaningful value is returned
    except requests.exceptions.RequestException as e:
        print(f"Error loading USD file: {e}")
        return {"status": "Error", "message": str(e)}

def activate_stream_livelink():
    payload = {"node_path": "/World/audio2face/StreamLivelink", "value": True}
    try:
        response = requests.post(f"{A2F_BASE_URL}/A2F/Exporter/ActivateStreamLivelink", json=payload)
        response.raise_for_status()
        result = response.json()
        print("Stream LiveLink activated successfully:", result)
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error activating Stream LiveLink: {e}")
        return {"status": "Error", "message": str(e)}


def set_stream_livelink_settings():
    payload = {
        "node_path": "/World/audio2face/StreamLivelink",
        "values": {"enable_audio_stream": True},
    }
    try:
        response = requests.post(f"{A2F_BASE_URL}/A2F/Exporter/SetStreamLivelinkSettings", json=payload)
        response.raise_for_status()
        result = response.json()
        print("Audio stream enabled successfully:", result)
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error enabling audio stream: {e}")
        return {"status": "Error", "message": str(e)}


def set_audio_looping():
    payload = {"a2f_player": "/World/audio2face/Player", "loop_audio": False}
    try:
        response = requests.post(f"{A2F_BASE_URL}/A2F/Player/SetLooping", json=payload)
        response.raise_for_status()
        result = response.json()
        print("Looping set to false:", result)
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error setting looping to false: {e}")
        return {"status": "Error", "message": str(e)}


# Initialize Gradio client
GRADIO_CLIENT = Client("lj1995/GPT-SoVITS-v2")

# Updated save_response_to_wav function
def save_response_to_wav(response_text, file_name):
    """
    Save the chatbot's response as a .wav file using GPT-SoVITS-v2 TTS API.
    """
    try:
        output_dir = AUDIO_DIR  # Use the existing AUDIO_DIR
        output_path = os.path.join(output_dir, file_name)

        # Call Gradio TTS API
        result = GRADIO_CLIENT.predict(
            ref_wav_path=handle_file(r"D:\Projects\MetaHUMANS\audio_sample_treim.wav"),  # No reference WAV file required
            prompt_text="",
            prompt_language="English",
            text=response_text,
            text_language="English",
            how_to_cut="Slice once every 4 sentences",
            top_k=15,
            top_p=1,
            temperature=1,
            ref_free=True,
            speed=1,
            if_freeze=False,
            inp_refs=[],  # Empty list for optional references
            api_name="/get_tts_wav"
        )

        # Save the generated WAV file to the desired location
        shutil.copy(result, output_path)

        print(f"Generated audio saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        return None


def send_audio_to_audio2face(file_name, response_counter):
    set_root_path_payload = {
        "a2f_player": "/World/audio2face/Player",
        "dir_path": AUDIO_DIR,
    }
    try:
        # Set root path
        response_root = requests.post(f"{A2F_BASE_URL}/A2F/Player/SetRootPath", json=set_root_path_payload)
        response_root.raise_for_status()

        # Set track
        set_track_payload = {
            "a2f_player": "/World/audio2face/Player",
            "file_name": f"response_{response_counter}.wav",
            "time_range": [0, -1],
        }
        response_track = requests.post(f"{A2F_BASE_URL}/A2F/Player/SetTrack", json=set_track_payload)
        response_track.raise_for_status()

        # Play track
        play_payload = {"a2f_player": "/World/audio2face/Player"}
        response_play = requests.post(f"{A2F_BASE_URL}/A2F/Player/Play", json=play_payload)
        response_play.raise_for_status()

        print(f"Audio sent successfully to Audio2Face for file: {file_name}")
        return {"status": "OK", "message": "Audio played successfully"}
    except requests.exceptions.RequestException as e:
        print(f"Error while interacting with Audio2Face API: {e}")
        return {"status": "Error", "message": str(e)}
def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Speak now!")
        try:
            audio = recognizer.listen(source)
            user_speech = recognizer.recognize_google(audio)
            print(f"You said: {user_speech}")
            return user_speech
        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None

def stop_audio2face_animation():
    """
    Sends a request to stop and reset the Audio2Face animation.
    """
    try:
        # Reset the time of the animation to 0
        reset_time_payload = {
            "a2f_player": "/World/audio2face/Player",
            "time": 0
        }
        reset_time_response = requests.post(f"{A2F_BASE_URL}/A2F/Player/SetTime", json=reset_time_payload)
        reset_time_response.raise_for_status()
        print("Animation time reset to 0:", reset_time_response.json())

        # Pause the animation
        pause_payload = {
            "a2f_player": "/World/audio2face/Player"
        }
        pause_response = requests.post(f"{A2F_BASE_URL}/A2F/Player/Pause", json=pause_payload)
        pause_response.raise_for_status()
        print("Animation paused successfully:", pause_response.json())

    except requests.exceptions.RequestException as e:
        print(f"Error while stopping Audio2Face animation: {e}")

def main():
    """
    Main function for the chatbot with sentiment analysis and Audio2Face integration.
    """
    check_api_status()
    load_usd_file()
    activate_stream_livelink()
    set_stream_livelink_settings()
    set_audio_looping()

    groq_api_key = os.environ["GROQ_API_KEY"]
    model = "llama3-8b-8192"

    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    system_prompt = "You are a friendly conversational chatbot. Give the response in 50 words"
    conversational_memory_length = 5

    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length, memory_key="chat_history", return_messages=True
    )

    response_counter = 1

    print("Press [1] to use voice input, or [2] to type your questions.")
    mode = input("Choose mode: ").strip()

    if mode not in ["1", "2"]:
        print("Invalid choice. Exiting...")
        return

    while True:
        user_question = None  # Initialize to avoid reference errors

        if mode == "1":
            user_question = get_speech_input()
            if user_question and user_question.lower() == "exit":
                print("Exiting... Goodbye!")
                stop_audio2face_animation()
                break
            sentiment = analyze_text_sentiment(user_question) if user_question else "Neutral"
            print(f"Sentiment (Text): {sentiment}")
        elif mode == "2":
            user_question = input("Ask a question (type 'exit' to quit): ").strip()
            if user_question.lower() == "exit":
                print("Exiting... Goodbye!")
                stop_audio2face_animation()
                break
            sentiment = analyze_text_sentiment(user_question) if user_question else "Neutral"
            print(f"Sentiment (Text): {sentiment}")

        if user_question:  # Ensure processing only happens if valid input is received
            conversation = LLMChain(
                llm=groq_chat,
                prompt=ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template("{human_input}"),
                    ]
                ),
                verbose=False,
                memory=memory,
            )

            try:
                # Generate chatbot response
                response = conversation.predict(human_input=user_question)
                print("Chatbot:", response)

                # Save response as audio and send to Audio2Face
                file_name = f"response_{response_counter}.wav"
                file_path = save_response_to_wav(response, file_name)
                send_audio_to_audio2face(file_name, response_counter)

                # Perform audio sentiment analysis
                chatbot_sentiment = analyze_text_sentiment(response)
                print(f"Sentiment (Chatbot[text]): {chatbot_sentiment}")

                # Generate emotion payload based on text and audio sentiment
                emotion_payload = generate_emotion_payload(sentiment,chatbot_sentiment)
                print(f"Emotion Payload: {emotion_payload}")

                # Send emotions to Audio2Face API
                response = requests.post(f"{A2F_BASE_URL}/A2F/A2E/SetEmotionByName", json=emotion_payload)
                response.raise_for_status()
                print("Emotions sent to Audio2Face successfully:", response.json())

                response_counter += 1

            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("No valid input received. Please try again.")
if __name__ == "__main__":
    main()