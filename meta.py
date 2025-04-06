# filename: app_server.py
import os
import requests
import speech_recognition as sr
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import numpy as np
import random
import json
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import yt_dlp
import vlc
import time
import sys
import ctypes
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
from threading import Thread, Lock
import cv2
#from fer import FER
# Assuming contact_db is not used based on the main loop logic in original meta.py
# from contact_db import init_db, add_contact, get_contact
import pywhatkit # Kept from original dependencies
import datetime # Kept from original dependencies
import re # Kept from original dependencies

# --- Flask Integration ---
from flask import Flask, request, jsonify, render_template

# --- Configuration & Setup ---

# !! IMPORTANT: Load API keys securely from environment variables or a config file !!
# Avoid hardcoding keys directly in the script.
# Example using .env file (create a pass.env file with GROQ_API_KEY='your_key')
# load_dotenv(dotenv_path="pass.env") # Uncomment if using a .env file
# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     print("❌ GROQ_API_KEY not found in environment variables or .env file. Exiting.")
#     sys.exit(1)
# os.environ["GROQ_API_KEY"] = groq_api_key # Set for Langchain

# --- OR --- (Less Secure - For quick testing ONLY)
# REMOVE THIS KEY BEFORE SHARING OR COMMITTING CODE
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    print("❌ Groq API Key is missing. Please set it.")
    sys.exit(1)
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# Directories and Files
BASE_DIR = r"D:/MetaHUMAN"
AUDIO_DIR = r"D:/MetaHUMAN/responses_output"
DOWNLOADS_DIR = os.path.join(BASE_DIR, "Downloads") # Music downloads
USER_PREFS_FILE = os.path.join(BASE_DIR, "user_prefs.json")
ERROR_LOG_FILE = os.path.join(BASE_DIR, "error_log.txt")
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")


# News Microservice URL (Ensure this service is running separately)
NEWS_SERVICE_URL = "http://localhost:5000/news"


# Email credentials (Load from .env file preferably)
load_dotenv(dotenv_path="pass.env") # Ensure pass.env exists with EMAIL_USER and EMAIL_PASS
email_user = os.getenv("EMAIL_USER")
email_pass = os.getenv("EMAIL_PASS")
smtp_host = "smtp.gmail.com"

# NVIDIA Audio2Face API base URL
A2F_BASE_URL = "http://localhost:8011" # Ensure Audio2Face is running

# VLC Installation Path (Update if necessary)
vlc_path = r"C:\Program Files\VideoLAN\VLC" # Common path, adjust if different
# Add VLC to PATH and set plugin path
os.environ["PATH"] += os.pathsep + vlc_path
os.environ["VLC_PLUGIN_PATH"] = os.path.join(vlc_path, "plugins")
dll_path = os.path.join(vlc_path, "libvlc.dll")
if os.path.exists(dll_path):
    try:
        ctypes.CDLL(dll_path)
        print("✅ VLC library successfully loaded!")
    except OSError as e:
        print(f"❌ Error loading VLC DLL: {e}")
        print("Ensure the VLC architecture (32-bit/64-bit) matches your Python architecture.")
        log_error(f"Error loading VLC DLL: {e}")
        # sys.exit(1) # Decide if VLC is critical enough to exit
    except Exception as e:
        print(f"❌ Unknown error loading VLC: {e}")
        log_error(f"Unknown error loading VLC: {e}")
        # sys.exit(1)
else:
    print(f"❌ VLC library not found at {dll_path}! Check your VLC installation path.")
    print(f"   Current VLC Path: {vlc_path}")
    # sys.exit(1) # Decide if VLC is critical

# --- Global Variables for Flask App State ---
user_prefs = {}
memory = None
conversation = None
groq_chat = None
tts_model = None
tts_tokenizer = None
vader_analyzer = None
response_counter = 1 # Initialize counter

# --- Utility Functions ---

def log_error(error_msg):
    """Logs errors to a file with timestamps."""
    try:
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}\n")
    except Exception as e:
        print(f"🚨 Critical Error: Could not write to log file {ERROR_LOG_FILE}: {e}")

def load_user_prefs():
    """Loads user preferences from a JSON file."""
    if os.path.exists(USER_PREFS_FILE):
        try:
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Error decoding user preferences file {USER_PREFS_FILE}: {e}. Using defaults.")
            log_error(f"JSON Decode Error in {USER_PREFS_FILE}: {e}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load user preferences from {USER_PREFS_FILE}: {e}. Using defaults.")
            log_error(f"Error loading user prefs {USER_PREFS_FILE}: {e}")
    return {"favorite_music_genre": "", "preferred_city": "London", "interests": []} # Default prefs

def save_user_prefs(prefs):
    """Saves user preferences to a JSON file."""
    try:
        with open(USER_PREFS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=4)
    except Exception as e:
        print(f"❌ Error saving user preferences to {USER_PREFS_FILE}: {e}")
        log_error(f"Error saving user prefs {USER_PREFS_FILE}: {e}")

def load_chat_history():
    """Loads chat history from a JSON file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Chat history file {CHAT_HISTORY_FILE} is corrupted or empty. Starting fresh.")
            log_error(f"Chat history JSON decode error in {CHAT_HISTORY_FILE}")
            return []
        except Exception as e:
             print(f"⚠️ Warning: Could not load chat history from {CHAT_HISTORY_FILE}: {e}. Starting fresh.")
             log_error(f"Error loading chat history {CHAT_HISTORY_FILE}: {e}")
             return []
    return []

def save_chat_history(chat_history_list):
    """Saves chat history (list of dicts) to a JSON file."""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_history_list, f, indent=4)
    except Exception as e:
        print(f"❌ Error saving chat history to {CHAT_HISTORY_FILE}: {e}")
        log_error(f"Error saving chat history {CHAT_HISTORY_FILE}: {e}")

def continuous_emotion_detection():
    """Runs facial emotion detection in a loop using the webcam."""
    global current_emotion, emotion_tracking_active
    try:
        detector = FER(mtcnn=True) # Or mtcnn=False if MTCNN causes issues
        cap = cv2.VideoCapture(0) # Use camera 0
        if not cap.isOpened():
            print("❌ Could not open webcam for continuous emotion tracking.")
            log_error("Could not open webcam for continuous tracking.")
            return

        print("✅ Continuous emotion detection started.")
        while emotion_tracking_active:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Could not read frame from webcam. Retrying...")
                log_error("Could not read frame from webcam during continuous detection.")
                time.sleep(1)
                continue # Try again

            try:
                # Analyze frame
                result = detector.detect_emotions(frame)

                if result and len(result) > 0:
                    # Get the dominant emotion from the first detected face
                    emotions = result[0]["emotions"]
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]

                    with emotion_lock:
                        # Update global emotion state only if confidence is reasonable (optional)
                        # if confidence > 0.4: # Example threshold
                        current_emotion = dominant_emotion
                        # else: current_emotion = "neutral" # Revert to neutral if low confidence

                    # Optional: print detected emotion
                    # print(f"Detected emotion: {dominant_emotion} ({confidence:.2f})")

                else:
                    # No face detected, assume neutral
                    with emotion_lock:
                        current_emotion = "neutral"
                    # print("No face detected. Assuming neutral.")

            except Exception as e:
                # Log errors during detection but keep the loop running
                print(f"❌ Error during single frame emotion detection: {e}")
                log_error(f"Error in single frame continuous emotion detection: {e}")
                with emotion_lock:
                    current_emotion = "neutral" # Reset to neutral on error

            time.sleep(1.0) # Check every 1 second to balance responsiveness and performance

    except ImportError:
         print("❌ FER or OpenCV not installed correctly. Emotion detection disabled.")
         log_error("FER/OpenCV import error. Continuous emotion detection disabled.")
    except Exception as e:
        print(f"❌ Unexpected error starting emotion detection: {e}")
        log_error(f"Fatal error starting continuous emotion detection: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            # cv2.destroyAllWindows() # Not needed in backend usually
        print("⏹️ Continuous emotion detection stopped.")

def get_location():
    """Fetches the user's approximate location based on IP address."""
    try:
        # Use a reliable IP geolocation service
        response = requests.get("http://ip-api.com/json/", timeout=5)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        if data.get('status') == 'success':
            return {
                "city": data.get("city", user_prefs.get("preferred_city", "London")),
                "country": data.get("country", "Unknown")
            }
        else:
            print(f"⚠️ Geolocation API did not return success: {data.get('message')}")
            log_error(f"Geolocation API error: Status {data.get('status')}, Message: {data.get('message')}")
            return {"city": user_prefs.get("preferred_city", "London"), "country": "UK"} # Fallback
    except requests.exceptions.Timeout:
        print("❌ Error fetching location: Request timed out.")
        log_error("Timeout error fetching location.")
        return {"city": user_prefs.get("preferred_city", "London"), "country": "UK"} # Fallback
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching location: {e}")
        log_error(f"Request error fetching location: {e}")
        return {"city": user_prefs.get("preferred_city", "London"), "country": "UK"} # Fallback
    except Exception as e:
        print(f"❌ Unexpected error fetching location: {e}")
        log_error(f"Unexpected error fetching location: {e}")
        return {"city": user_prefs.get("preferred_city", "London"), "country": "UK"} # Fallback


def get_weather(city):
    """Fetches weather information for a given city using wttr.in."""
    if not city:
        return "Please specify a city for the weather."
    try:
        # Use a simple format: Condition + Temperature
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url, timeout=7) # Increased timeout slightly
        response.raise_for_status()
        weather_data = response.text.strip()
        if "Unknown location" in weather_data or not weather_data :
             return f"Sorry, I couldn't find weather data for {city}."
        return weather_data
    except requests.exceptions.Timeout:
        print(f"❌ Error fetching weather for {city}: Request timed out.")
        log_error(f"Timeout error fetching weather for {city}.")
        return f"Unable to fetch weather data for {city} right now (timeout)."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching weather for {city}: {e}")
        log_error(f"Request error fetching weather for {city}: {e}")
        # Check for common 404 or other specific issues if possible
        return f"Unable to fetch weather data for {city} due to a network issue."
    except Exception as e:
        print(f"❌ Unexpected error fetching weather for {city}: {e}")
        log_error(f"Unexpected error fetching weather for {city}: {e}")
        return f"An unexpected error occurred while getting weather for {city}."

def download_music(song_name):
    """Downloads music from YouTube using yt-dlp."""
    print(f"🔍 Searching and downloading: '{song_name}'")
    # Sanitize song name for filesystem
    safe_song_name = re.sub(r'[\\/*?:"<>|]', "", song_name) # Remove invalid chars
    safe_song_name = "_".join(safe_song_name.split()) # Replace spaces with underscores
    file_name = f"{safe_song_name[:100]}" # Limit filename length
    # Output path uses the constant DOWNLOADS_DIR
    file_path_template = os.path.join(DOWNLOADS_DIR, file_name) # Template without extension
    final_file_path = file_path_template + ".mp3" # Expected final path

    print(f"DEBUG: Target download directory: {DOWNLOADS_DIR}")
    print(f"DEBUG: Sanitized filename base: {file_name}")
    print(f"DEBUG: Expected final file path: {final_file_path}")

    if os.path.exists(final_file_path):
        print(f"🎵 Song already downloaded at: {final_file_path}")
        return final_file_path

    ydl_opts = {
        'format': 'bestaudio/best',        # Get the best audio quality
        'outtmpl': file_path_template + '.%(ext)s', # Use template, yt-dlp adds extension
        'noplaylist': True,                # Don't download playlists if a link resolves to one
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',       # Convert to MP3
            'preferredquality': '128',     # Standard quality MP3
        }],
        'quiet': False,                    # Show yt-dlp output
        'default_search': 'ytsearch1',     # Search YouTube and take the first result
        'nocheckcertificate': True,       # Sometimes needed for network issues
        # 'verbose': True,                 # Uncomment for detailed logs from yt-dlp
        # 'progress_hooks': [lambda d: print(f"Download progress: {d.get('_percent_str', 'N/A')} of {d.get('_total_bytes_str', 'N/A')}") if d.get('status') == 'downloading' else None],
    }

    try:
        # Ensure FFmpeg is available or provide path if needed
        # ydl_opts['ffmpeg_location'] = '/path/to/your/ffmpeg'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Starting yt-dlp download...")
            ydl.download([f"ytsearch1:{song_name}"]) # Search and download
            print("yt-dlp download process finished.")

        # Check if the MP3 file exists after processing
        if os.path.exists(final_file_path):
            print(f"✅ Downloaded and converted to: {final_file_path}")
            return final_file_path
        else:
            # Check if intermediate file exists (e.g., .webm, .m4a) before conversion failed
            possible_intermediate = next((os.path.join(DOWNLOADS_DIR, f) for f in os.listdir(DOWNLOADS_DIR) if f.startswith(file_name) and f != final_file_path), None)
            if possible_intermediate:
                 print(f"⚠️ Download completed but conversion to MP3 might have failed. Intermediate file found: {possible_intermediate}")
                 log_error(f"Music download for '{song_name}' completed but MP3 conversion failed. Check FFmpeg. Intermediate: {possible_intermediate}")
                 # Optionally try to return the intermediate file or signal conversion error
                 # return possible_intermediate # If player supports it
                 raise FileNotFoundError(f"MP3 conversion failed for {song_name}. Check FFmpeg.")
            else:
                 print(f"❌ Download failed. Expected file not found: {final_file_path}")
                 raise FileNotFoundError(f"Downloaded file not found after yt-dlp process: {final_file_path}")

    except yt_dlp.utils.DownloadError as e:
        print(f"❌ yt-dlp Download Error for '{song_name}': {e}")
        log_error(f"yt-dlp Download Error for '{song_name}': {e}")
        raise # Re-raise the specific error
    except FileNotFoundError as e:
         print(f"❌ File Not Found Error after download attempt for '{song_name}': {e}")
         log_error(f"FileNotFoundError after download for '{song_name}': {e}")
         raise # Re-raise
    except Exception as e:
        # Catch potential errors during postprocessing (ffmpeg)
        print(f"❌ Unexpected Error during music download/processing for '{song_name}': {e}")
        log_error(f"Unexpected error downloading music '{song_name}': {e}")
        raise # Re-raise the error to be handled by the caller

def play_music_threaded(file_path):
    """Plays music in a separate thread using VLC without blocking the main thread."""
    global current_emotion # Access global emotion state
    if not file_path or not os.path.exists(file_path):
        print(f"❌ Audio file not found for playback thread: {file_path}")
        log_error(f"Audio file not found for playback thread: {file_path}")
        return

    def playback_task():
        player = None
        try:
            # Ensure VLC instance is created within the thread if needed, or use a shared instance carefully
            vlc_instance = vlc.Instance()
            player = vlc_instance.media_player_new()
            media = vlc_instance.media_new(file_path)
            player.set_media(media)

            if not player:
                print("❌ Failed to initialize VLC player instance in thread.")
                log_error("Failed to initialize VLC player instance in thread.")
                return

            print(f"🎵 Attempting to play music in background: {os.path.basename(file_path)}")
            player.play()

            # Wait briefly for playback to start
            time.sleep(1)
            if not player.is_playing():
                 # Check state more closely if not playing
                 state = player.get_state()
                 print(f"⚠️ Playback did not start for {os.path.basename(file_path)}. VLC State: {state}")
                 log_error(f"Playback did not start for {os.path.basename(file_path)}. VLC State: {state}")
                 # Maybe release resources if playback failed immediately
                 # player.release()
                 # return

            # Loop to monitor playback status and emotion
            while True:
                state = player.get_state()
                # print(f"DEBUG: Playback state for {os.path.basename(file_path)}: {state}") # Verbose logging

                if state in [vlc.State.Ended, vlc.State.Stopped, vlc.State.Error]:
                    print(f"🎵 Playback finished or stopped for {os.path.basename(file_path)}. Final state: {state}")
                    if state == vlc.State.Error:
                         log_error(f"VLC Error during playback for: {os.path.basename(file_path)}")
                    break # Exit the monitoring loop

                # Check for negative emotion and stop playback if detected
                with emotion_lock:
                    user_emotion = current_emotion
                if user_emotion in ["sad", "angry", "disgust"] and player.is_playing():
                    print(f"⚠️ Detected negative emotion ({user_emotion}), stopping music: {os.path.basename(file_path)}")
                    log_error(f"Stopping music playback due to detected emotion: {user_emotion}")
                    player.stop()
                    # Note: Cannot easily trigger a *new* bot response from this thread back to the user via Flask
                    break # Exit loop after stopping

                time.sleep(1.5) # Check status less frequently to reduce overhead

        except vlc.VLCException as e:
             print(f"❌ VLC Exception in playback thread for {os.path.basename(file_path)}: {e}")
             log_error(f"VLCException in playback thread for {os.path.basename(file_path)}: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error in playback thread for {os.path.basename(file_path)}: {e}")
            log_error(f"Unexpected error in playback thread: {e}")
        finally:
            # Ensure resources are released
            if player:
                try:
                    if player.is_playing():
                        player.stop()
                    player.release()
                    print(f"Released VLC player for: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️ Error releasing VLC player: {e}")
                    log_error(f"Error releasing VLC player: {e}")
            # Optionally remove the file after playback attempt (consider if reuse is desired)
            # if os.path.exists(file_path):
            #     try:
            #         os.remove(file_path)
            #         print(f"🗑️ Removed played file: {file_path}")
            #     except OSError as e:
            #         print(f"⚠️ Could not remove file {file_path}: {e}")
            #         log_error(f"Error removing played file {file_path}: {e}")

    # Start the playback task in a separate daemon thread
    # Daemon threads exit automatically when the main program exits
    thread = Thread(target=playback_task, daemon=True, name=f"MusicPlayer_{os.path.basename(file_path)}")
    thread.start()
    print(f"Started background playback thread for: {os.path.basename(file_path)}")


def send_email_with_attachments(to_email, subject, body_content, attachments=[]):
    """Sends an email using configured credentials."""
    if not email_user or not email_pass:
        msg = "Email credentials (EMAIL_USER, EMAIL_PASS) are not configured. Cannot send email."
        print(f"❌ {msg}")
        log_error(msg)
        return msg

    try:
        msg = EmailMessage()
        msg["From"] = email_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body_content)

        # Handle attachments (assuming attachments are file paths)
        for file_path_str in attachments:
             if isinstance(file_path_str, str) and os.path.exists(file_path_str):
                 try:
                     with open(file_path_str, "rb") as file:
                         file_data = file.read()
                         file_name = os.path.basename(file_path_str)
                         # Determine MIME type or use generic octet-stream
                         # import mimetypes
                         # ctype, encoding = mimetypes.guess_type(file_path_str)
                         # if ctype is None or encoding is not None:
                         #     ctype = 'application/octet-stream'
                         # maintype, subtype = ctype.split('/', 1)
                         # msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)
                         msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)
                 except Exception as e:
                      print(f"⚠️ Warning: Could not attach file {file_path_str}: {e}")
                      log_error(f"Error attaching file {file_path_str} to email: {e}")
             else:
                 print(f"⚠️ Warning: Attachment path invalid or not found: {file_path_str}")


        print(f"📧 Attempting to send email to {to_email} via {smtp_host}...")
        # Use Gmail's SMTP server with SSL
        with smtplib.SMTP_SSL(smtp_host, 465) as server:
            server.login(email_user, email_pass)
            server.send_message(msg)
            success_msg = f"✅ Email sent successfully to {to_email}."
            print(success_msg)
            return success_msg
    except smtplib.SMTPAuthenticationError:
        error_msg = "❌ Error sending email: Authentication failed. Check email/password or 'less secure app access' settings."
        print(error_msg)
        log_error("SMTP Authentication Error sending email.")
        return error_msg
    except smtplib.SMTPException as e:
        error_msg = f"❌ Error sending email (SMTP Error): {e}"
        print(error_msg)
        log_error(f"SMTP Error sending email: {e}")
        return error_msg
    except Exception as e:
        error_msg = f"❌ Unexpected error sending email: {e}"
        print(error_msg)
        log_error(f"Unexpected error sending email: {e}")
        return f"Error sending email: {str(e)}"


def get_news(query):
    """Fetches news articles from the news microservice."""
    if not query:
        return "What topic would you like the news about?"
    try:
        print(f"📰 Fetching news from microservice for query: '{query}'")
        response = requests.get(f"{NEWS_SERVICE_URL}?query={query}", timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "success" and data.get("articles"):
            articles = data["articles"][:3] # Limit to top 3 articles
            news_summary = f"Here’s the latest news I found on '{query}':\n"
            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No Title')
                source = article.get('source', 'Unknown Source')
                # Optionally add description or URL if available and desired
                # description = article.get('description', '')
                # url = article.get('url', '')
                news_summary += f"{i}. {title} (Source: {source}).\n"
            return news_summary.strip()
        elif data.get("status") == "error":
             error_msg = data.get("message", "Unknown error from news service")
             print(f"❌ News service returned an error: {error_msg}")
             log_error(f"News service error for query '{query}': {error_msg}")
             return f"Sorry, the news service had an issue fetching news for '{query}': {error_msg}"
        else:
            print(f"📰 No news articles found for '{query}'.")
            return f"Sorry, I couldn’t find any recent news articles matching '{query}' right now."

    except requests.exceptions.Timeout:
        print(f"❌ Error fetching news for '{query}': Request timed out.")
        log_error(f"Timeout fetching news from microservice for '{query}'.")
        return f"Unable to fetch news for '{query}' right now (timeout contacting service)."
    except requests.exceptions.ConnectionError:
        print(f"❌ Error fetching news: Could not connect to the News Service at {NEWS_SERVICE_URL}.")
        log_error(f"ConnectionError contacting News Service at {NEWS_SERVICE_URL}.")
        return "Sorry, I can't connect to the news service at the moment. Please ensure it's running."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching news for '{query}': {e}")
        log_error(f"Request error fetching news for '{query}': {e}")
        return f"Unable to fetch news for '{query}' due to a network or service error."
    except Exception as e:
        print(f"❌ Unexpected error fetching news for '{query}': {e}")
        log_error(f"Unexpected error fetching news for '{query}': {e}")
        return f"An unexpected error occurred while getting news about '{query}'."


def analyze_text_sentiment(text):
    """Analyzes text sentiment using VADER."""
    if not text or not vader_analyzer:
        return "Neutral"
    try:
        sentiment_score = vader_analyzer.polarity_scores(text)
        compound = sentiment_score['compound']
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print(f"❌ Error analyzing text sentiment: {e}")
        log_error(f"Error analyzing text sentiment: {e}")
        return "Neutral" # Default to Neutral on error

def analyze_audio_sentiment(audio_path):
    """Placeholder/Basic audio sentiment analysis based on energy."""
    # Note: This is a very basic and likely inaccurate method.
    # Proper audio sentiment analysis requires more sophisticated models.
    if not os.path.exists(audio_path):
        return "Neutral"
    try:
        [fs, x] = audioBasicIO.read_audio_file(audio_path)
        if x.ndim > 1:
             x = audioBasicIO.stereo_to_mono(x) # Convert to mono if stereo

        # Basic energy feature - highly experimental for sentiment
        # Use appropriate window and step sizes based on sampling rate (fs)
        window_size = 0.050 # 50 ms window
        step_size = 0.025  # 25 ms step
        features, _ = ShortTermFeatures.feature_extraction(x, fs, window_size * fs, step_size * fs)
        # Feature 1 is typically energy
        if features.shape[0] > 1:
             energy = np.mean(features[1, :]) # Mean energy across frames
             # Thresholds are arbitrary and need tuning
             if energy > 0.01: return "Positive"
             elif energy < 0.005: return "Negative"
             else: return "Neutral"
        else:
             print("⚠️ Could not extract sufficient features for audio sentiment.")
             return "Neutral"

    except Exception as e:
        print(f"❌ Error analyzing audio sentiment for {audio_path}: {e}")
        log_error(f"Error analyzing audio sentiment for {audio_path}: {e}")
        return "Neutral" # Default on error


def randomize_emotion(base, variation=0.1):
    """Generates a randomized emotion value around a base."""
    return round(max(0.0, min(1.0, random.uniform(base - variation, base + variation))), 2)

def generate_emotion_payload(user_sentiment_or_emotion, chatbot_sentiment):
    """Generates the emotion payload for Audio2Face based on sentiments/emotions."""
    # Base emotion dictionary
    emotions = {
        "amazement": 0.0, "anger": 0.0, "cheekiness": 0.0, "disgust": 0.0, "fear": 0.0,
        "grief": 0.0, "joy": 0.0, "outofbreath": 0.0, "pain": 0.0, "sadness": 0.0,
        # Add other potential A2F emotions if available/needed
    }

    # Determine input condition (can be text sentiment 'Positive'/'Negative'/'Neutral'
    # or a detected facial emotion like 'happy', 'sad', 'angry')
    input_condition = user_sentiment_or_emotion.lower()

    # --- Emotion Mapping Logic ---
    # This logic can be significantly expanded and refined.

    # Positive User Input
    if input_condition in ["positive", "happy", "joy", "surprise"]: # Surprise mapped to positive reaction
        if chatbot_sentiment == "Positive":
            emotions["joy"] = randomize_emotion(0.8, 0.2)
            emotions["cheekiness"] = randomize_emotion(0.3, 0.15) # Add a bit of playful cheekiness
        elif chatbot_sentiment == "Neutral":
            emotions["joy"] = randomize_emotion(0.6, 0.2)
            # Maybe slight cheekiness if neutral response to positive input
            emotions["cheekiness"] = randomize_emotion(0.1, 0.1)
        else: # Negative Chatbot Response to Positive User -> Maybe confusion/slight sadness?
            emotions["sadness"] = randomize_emotion(0.3, 0.15)
            # emotions["fear"] = randomize_emotion(0.2, 0.1) # Or slight fear/concern

    # Neutral User Input
    elif input_condition == "neutral":
        if chatbot_sentiment == "Positive":
            emotions["joy"] = randomize_emotion(0.5, 0.2)
            emotions["cheekiness"] = randomize_emotion(0.4, 0.2)
        elif chatbot_sentiment == "Neutral":
            # Keep mostly neutral, maybe slight cheekiness for engagement
             emotions["cheekiness"] = randomize_emotion(0.2, 0.1)
        else: # Negative Chatbot Response to Neutral User
            emotions["sadness"] = randomize_emotion(0.5, 0.2)
            emotions["grief"] = randomize_emotion(0.3, 0.15) # Use grief slightly

    # Negative User Input
    elif input_condition in ["negative", "sad", "angry", "disgust", "fear"]:
        if chatbot_sentiment == "Positive": # Positive Chatbot Response to Negative User -> Empathetic/Trying to cheer up
            emotions["joy"] = randomize_emotion(0.4, 0.2) # Gentle joy/encouragement
            emotions["sadness"] = randomize_emotion(0.3, 0.15) # Acknowledging sadness
        elif chatbot_sentiment == "Neutral": # Neutral Chatbot Response -> Calm, supportive sadness
            emotions["sadness"] = randomize_emotion(0.6, 0.2)
            emotions["grief"] = randomize_emotion(0.4, 0.2)
        else: # Negative Chatbot Response -> Mirroring negativity (use carefully)
             if input_condition == "angry":
                  emotions["anger"] = randomize_emotion(0.6, 0.2)
                  emotions["disgust"] = randomize_emotion(0.3, 0.1)
             elif input_condition == "disgust":
                  emotions["disgust"] = randomize_emotion(0.7, 0.2)
                  emotions["anger"] = randomize_emotion(0.2, 0.1)
             elif input_condition == "fear":
                  emotions["fear"] = randomize_emotion(0.6, 0.2)
                  emotions["sadness"] = randomize_emotion(0.3, 0.1)
             else: # Default to sadness/grief for general negative
                 emotions["sadness"] = randomize_emotion(0.7, 0.2)
                 emotions["grief"] = randomize_emotion(0.5, 0.2)

    # Construct the final payload for the A2F API
    payload = {
        "a2f_instance": "/World/audio2face/CoreFullface", # Ensure this path matches your A2F setup
        "emotions": emotions,
    }
    # print(f"DEBUG: Generated Emotion Payload: {payload}") # Optional debug print
    return payload


def check_api_status(url=A2F_BASE_URL):
    """Checks if the Audio2Face API is reachable."""
    try:
        response = requests.get(f"{url}/status", timeout=3) # Short timeout for status check
        response.raise_for_status()
        print(f"✅ Audio2Face API Status at {url}: OK ({response.json()})")
        return True
    except requests.exceptions.Timeout:
        print(f"⚠️ Warning: Timeout connecting to Audio2Face API at {url}. Is it running?")
        log_error(f"Timeout connecting to A2F API at {url}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"⚠️ Warning: Connection refused by Audio2Face API at {url}. Is it running and accessible?")
        log_error(f"Connection refused by A2F API at {url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Warning: Error checking Audio2Face API status at {url}: {e}")
        log_error(f"Error checking A2F API status at {url}: {e}")
        return False


# --- Optional A2F Setup Functions (Call during initialization if needed) ---

def load_usd_file(file_path=r"D:\FF_Recieptionist_Backend\Models\mark.usd"): # Example Path
    """Loads a specific USD file into Audio2Face."""
    if not check_api_status(): return {"status": "Error", "message": "A2F API not reachable"}
    payload = {"file_name": file_path}
    try:
        response = requests.post(f"{A2F_BASE_URL}/A2F/USD/Load", json=payload, timeout=10) # Longer timeout for loading
        response.raise_for_status()
        print(f"✅ USD file '{file_path}' loaded successfully: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Warning: Error loading USD file '{file_path}': {e}")
        log_error(f"Error loading A2F USD file '{file_path}': {e}")
        return {"status": "Error", "message": str(e)}

def activate_stream_livelink(node_path="/World/audio2face/StreamLivelink", activate=True):
    """Activates or deactivates the StreamLivelink exporter in Audio2Face."""
    if not check_api_status(): return {"status": "Error", "message": "A2F API not reachable"}
    payload = {"node_path": node_path, "value": activate}
    action = "Activating" if activate else "Deactivating"
    try:
        response = requests.post(f"{A2F_BASE_URL}/A2F/Exporter/ActivateStreamLivelink", json=payload, timeout=5)
        response.raise_for_status()
        print(f"✅ Stream LiveLink at '{node_path}' {action.lower()} successful: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Warning: Error {action.lower()} Stream LiveLink at '{node_path}': {e}")
        log_error(f"Error {action.lower()} A2F Stream LiveLink at '{node_path}': {e}")
        return {"status": "Error", "message": str(e)}

def set_stream_livelink_settings(node_path="/World/audio2face/StreamLivelink", settings={"enable_audio_stream": False}):
    """Configures settings for the StreamLivelink exporter."""
    if not check_api_status(): return {"status": "Error", "message": "A2F API not reachable"}
    payload = {
        "node_path": node_path,
        "values": settings,
    }
    try:
        response = requests.post(f"{A2F_BASE_URL}/A2F/Exporter/SetStreamLivelinkSettings", json=payload, timeout=5)
        response.raise_for_status()
        print(f"✅ Stream LiveLink settings updated for '{node_path}': {settings} -> {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Warning: Error setting Stream LiveLink settings for '{node_path}': {e}")
        log_error(f"Error setting A2F Stream LiveLink settings for '{node_path}': {e}")
        return {"status": "Error", "message": str(e)}

def set_audio_looping(player_path="/World/audio2face/Player", loop=False):
    """Sets the audio looping state in the Audio2Face player."""
    if not check_api_status(): return {"status": "Error", "message": "A2F API not reachable"}
    payload = {"a2f_player": player_path, "loop_audio": loop}
    try:
        response = requests.post(f"{A2F_BASE_URL}/A2F/Player/SetLooping", json=payload, timeout=5)
        response.raise_for_status()
        print(f"✅ Audio looping set to {loop} for player '{player_path}': {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Warning: Error setting audio looping for player '{player_path}': {e}")
        log_error(f"Error setting A2F audio looping for player '{player_path}': {e}")
        return {"status": "Error", "message": str(e)}

# --- Core TTS and A2F Interaction Functions ---

def save_response_to_wav(response_text, file_name):
    """Generates audio from text using the loaded TTS model and saves it to a WAV file."""
    global tts_model, tts_tokenizer # Use globally loaded models
    if not tts_model or not tts_tokenizer:
        print("❌ TTS model not loaded. Cannot generate audio.")
        log_error("TTS model not loaded, save_response_to_wav failed.")
        return None
    if not response_text:
        print("⚠️ Warning: Empty response text provided for TTS.")
        return None

    output_path = os.path.join(AUDIO_DIR, file_name)
    try:
        print(f"🎙️ Generating TTS audio for: '{response_text[:50]}...'")
        inputs = tts_tokenizer(response_text, return_tensors="pt")
        with torch.no_grad():
            # Generate waveform
            audio_waveform = tts_model(**inputs).waveform
            # Ensure it's a 1D numpy array on CPU
            audio_numpy = audio_waveform.squeeze().cpu().numpy()

        # Check if output is valid
        if audio_numpy is None or audio_numpy.size == 0:
             raise ValueError("TTS model generated empty audio output.")

        # Get sampling rate from model config
        sampling_rate = tts_model.config.sampling_rate

        # Save using soundfile
        sf.write(output_path, audio_numpy, samplerate=sampling_rate)
        print(f"✅ Generated audio saved to: {output_path}")
        return output_path

    except AttributeError as e:
         print(f"❌ Error accessing TTS model attributes (maybe model loading failed?): {e}")
         log_error(f"AttributeError during TTS generation (model loaded?): {e}")
         return None
    except Exception as e:
        print(f"❌ Unexpected Error in TTS generation or saving for '{file_name}': {e}")
        log_error(f"Unexpected error in TTS generation/saving '{file_name}': {e}")
        # Clean up potentially corrupted file if it exists
        if os.path.exists(output_path):
             try: os.remove(output_path)
             except OSError: pass
        return None

def send_audio_to_audio2face(file_name, current_response_counter):
    """Sends the generated audio file to Audio2Face for playback and animation."""
    if not check_api_status():
         print("⚠️ Skipping A2F audio send: API not reachable.")
         return {"status": "Error", "message": "A2F API not reachable"}

    audio_file_path = os.path.join(AUDIO_DIR, file_name)
    if not os.path.exists(audio_file_path):
        print(f"❌ Cannot send audio to A2F: File not found at {audio_file_path}")
        log_error(f"Audio file not found for A2F send: {audio_file_path}")
        return {"status": "Error", "message": "Audio file not found"}

    player_path = "/World/audio2face/Player" # Default player path

    try:
        # 1. Set the root path where A2F should look for audio files
        set_root_path_payload = {
            "a2f_player": player_path,
            "dir_path": AUDIO_DIR, # Send the directory path
        }
        print(f"➡️ Setting A2F root path to: {AUDIO_DIR}")
        response_root = requests.post(f"{A2F_BASE_URL}/A2F/Player/SetRootPath", json=set_root_path_payload, timeout=5)
        response_root.raise_for_status()
        # print(f"DEBUG: SetRootPath response: {response_root.json()}") # Optional debug

        # 2. Set the specific audio track to be played (using relative filename)
        set_track_payload = {
            "a2f_player": player_path,
            "file_name": file_name, # Just the filename, relative to root path
            "time_range": [0, -1], # Play entire file (0 to end)
        }
        print(f"➡️ Setting A2F track to: {file_name}")
        response_track = requests.post(f"{A2F_BASE_URL}/A2F/Player/SetTrack", json=set_track_payload, timeout=5)
        response_track.raise_for_status()
        # print(f"DEBUG: SetTrack response: {response_track.json()}") # Optional debug

        # 3. Start playback
        play_payload = {"a2f_player": player_path}
        print(f"➡️ Sending Play command to A2F for: {file_name}")
        response_play = requests.post(f"{A2F_BASE_URL}/A2F/Player/Play", json=play_payload, timeout=5)
        response_play.raise_for_status()
        # print(f"DEBUG: Play response: {response_play.json()}") # Optional debug

        print(f"✅ Audio '{file_name}' sent successfully to Audio2Face.")
        return {"status": "OK", "message": "Audio sent and play command issued."}

    except requests.exceptions.Timeout:
         print(f"❌ Timeout error while interacting with Audio2Face API for {file_name}.")
         log_error(f"Timeout error interacting with A2F API for {file_name}.")
         return {"status": "Error", "message": "Timeout interacting with A2F API"}
    except requests.exceptions.RequestException as e:
        # Log the specific step that failed if possible
        step = "SetRootPath" # Assume first step initially
        if 'response_root' in locals() and response_root.ok: step = "SetTrack"
        if 'response_track' in locals() and response_track.ok: step = "Play"
        print(f"❌ Error interacting with Audio2Face API during '{step}' for {file_name}: {e}")
        # Log response body if available and useful
        response_content = e.response.text if e.response else "No response body"
        print(f"   Response Body: {response_content[:500]}") # Limit log size
        log_error(f"A2F API RequestException during '{step}' for {file_name}: {e}. Response: {response_content[:500]}")
        return {"status": "Error", "message": f"A2F API Error during {step}: {e}"}
    except Exception as e:
         print(f"❌ Unexpected error sending audio to Audio2Face for {file_name}: {e}")
         log_error(f"Unexpected error sending audio to A2F for {file_name}: {e}")
         return {"status": "Error", "message": f"Unexpected A2F error: {e}"}


def stop_audio2face_animation(player_path="/World/audio2face/Player"):
    """Attempts to stop the current animation/playback in Audio2Face."""
    if not check_api_status():
         print("⚠️ Skipping A2F stop: API not reachable.")
         return False

    try:
        # Optional: Reset time to 0 first
        reset_time_payload = {"a2f_player": player_path, "time": 0}
        try:
            reset_time_response = requests.post(f"{A2F_BASE_URL}/A2F/Player/SetTime", json=reset_time_payload, timeout=3)
            reset_time_response.raise_for_status()
            print("⏹️ A2F Animation time reset to 0.")
        except requests.exceptions.RequestException as e:
            # Don't treat failure to reset time as critical, pausing is more important
            print(f"⚠️ Warning: Could not reset A2F animation time: {e}")
            # log_error(f"Could not reset A2F time: {e}") # Optional logging

        # Send Pause command
        pause_payload = {"a2f_player": player_path}
        print("⏹️ Sending Pause command to A2F...")
        pause_response = requests.post(f"{A2F_BASE_URL}/A2F/Player/Pause", json=pause_payload, timeout=5)
        pause_response.raise_for_status()
        print("✅ A2F Animation paused successfully.")
        return True

    except requests.exceptions.Timeout:
        print("❌ Timeout error while stopping Audio2Face animation.")
        log_error("Timeout error stopping A2F animation.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error while stopping Audio2Face animation: {e}")
        log_error(f"Error stopping Audio2Face animation: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error stopping Audio2Face animation: {e}")
        log_error(f"Unexpected error stopping A2F animation: {e}")
        return False


# --- Initialization Function ---
def initialize_app():
    """Performs all necessary initializations when the Flask app starts."""
    global user_prefs, memory, conversation, groq_chat, response_counter, tts_model, tts_tokenizer, vader_analyzer

    print("\n" + "="*30 + " Initializing MetaHuman Assistant Server " + "="*30)

    # 1. Load User Preferences
    print("Loading user preferences...")
    user_prefs = load_user_prefs()
    print(f"User preferences loaded: {user_prefs}")

    # 2. Initialize VADER Sentiment Analyzer
    print("Initializing VADER sentiment analyzer...")
    try:
        vader_analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER initialized.")
    except Exception as e:
         print(f"❌ Failed to initialize VADER: {e}")
         log_error(f"Failed to initialize VADER: {e}")
         # Decide if this is critical
         # sys.exit(1)

    # 3. Load TTS Model (MMS-TTS)
    print("Loading TTS model (facebook/mms-tts-eng)... This may take some time.")
    try:
        # Explicitly use auth token if needed (replace with your actual token if required by HF)
        # auth_token = os.getenv("HUGGINGFACE_TOKEN", "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") # Example token, use yours
        # if not auth_token: print("⚠️ Hugging Face token not found, model download might fail if private.")

        tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng") #, use_auth_token=auth_token)
        tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng") #, use_auth_token=auth_token)
        print("✅ TTS model and tokenizer loaded successfully.")
    except OSError as e:
         print(f"❌ Error loading TTS model: {e}. Likely network issue or model name incorrect.")
         log_error(f"OSError loading TTS model: {e}")
         print("   Please check internet connection and model availability on Hugging Face Hub.")
         sys.exit(1) # TTS is critical for audio output
    except Exception as e:
        print(f"❌ Critical Error loading TTS model: {e}")
        log_error(f"Critical Error loading TTS model: {e}")
        sys.exit(1) # Exit if TTS fails

    # 4. Load Chat History & Initialize Langchain Memory
    print("Loading chat history and initializing Langchain memory...")
    try:
        initial_history_raw = load_chat_history()
        initial_history_messages = []
        for msg_data in initial_history_raw:
            msg_type = msg_data.get('type')
            content = msg_data.get('content', '')
            if msg_type == 'human':
                initial_history_messages.append(HumanMessage(content=content))
            elif msg_type == 'ai':
                 initial_history_messages.append(AIMessage(content=content))
            # Handle other potential types like 'system' if necessary
            # elif msg_type == 'system':
            #      initial_history_messages.append(SystemMessage(content=content))
            else:
                 print(f"⚠️ Skipping message with unknown type '{msg_type}' during history loading.")

        conversational_memory_length = 10 # Keep N turns (N human + N AI messages)
        memory = ConversationBufferWindowMemory(
            k=conversational_memory_length,
            memory_key="chat_history",
            return_messages=True,
            # Optionally load initial messages here if format allows
            # chat_memory=ChatMessageHistory(messages=initial_history_messages) # More direct way
        )
        # Add messages manually if not using direct initialization
        for msg in initial_history_messages:
             memory.chat_memory.add_message(msg)

        print(f"✅ Langchain memory initialized. Loaded {len(memory.chat_memory.messages)} messages from history.")

    except Exception as e:
        print(f"❌ Error loading chat history or initializing memory: {e}. Starting with empty memory.")
        log_error(f"Error initializing Langchain memory: {e}")
        # Fallback to empty memory
        conversational_memory_length = 10
        memory = ConversationBufferWindowMemory(
            k=conversational_memory_length, memory_key="chat_history", return_messages=True
        )

    # 5. Initialize Groq Chat LLM
    print("Initializing Groq Chat LLM (llama3-8b-8192)...")
    try:
        groq_api_key_used = os.environ["GROQ_API_KEY"] # Get key set earlier
        model_name = "llama3-8b-8192" # Or choose another available model
        groq_chat = ChatGroq(groq_api_key=groq_api_key_used, model_name=model_name)
        # Test connection briefly (optional)
        # groq_chat.invoke("Hello!")
        print("✅ Groq Chat LLM initialized successfully.")
    except KeyError:
         print("❌ Critical Error: GROQ_API_KEY environment variable not set.")
         log_error("GROQ_API_KEY not set during Groq Chat initialization.")
         sys.exit(1)
    except Exception as e:
        print(f"❌ Critical Error initializing Groq Chat: {e}")
        log_error(f"Critical Error initializing Groq Chat: {e}")
        sys.exit(1) # LLM is critical

    # 6. Setup Langchain Conversation Runnable
    print("Setting up Langchain conversation runnable...")
    system_prompt = (
        "You are 'Meta', a smart, emotionally aware, and humorous personal assistant. "
        "Your goal is to be helpful, engaging, and adaptive to the user's mood, which you subtly infer from their input and potentially facial expressions (provided externally). "
        "Handle tasks like weather updates, playing music (by initiating downloads), sending emails (confirm first!), getting news summaries, and general conversation. "
        "Keep responses concise (under 60 words) unless details are necessary. "
        "Sound natural: use contractions, light humor, occasional emojis (use sparingly and appropriately), and conversational fillers. "
        "If the user seems down (sad, angry), be more supportive and calm. If they seem happy, be more upbeat and playful. "
        "If asked to perform an action like sending email or playing specific music, briefly confirm the core details. "
        "Example: 'Okay, sending an email to [recipient] about [subject]?' or 'Got it, downloading '[song name]' now!'"
        "You have access to real-time weather and news. You can trigger music downloads and playback. You can attempt to send emails."
        "Current user emotion (estimated externally): {user_emotion}" # Placeholder for dynamic injection if possible
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt), # System prompt provides context
            MessagesPlaceholder(variable_name="chat_history"), # Placeholder for past messages
            HumanMessagePromptTemplate.from_template("{human_input}"), # Template for the current user input
        ]
    )
    conversation = RunnableSequence(prompt_template | groq_chat)
    print("✅ Langchain conversation runnable created.")

    # 7. Initialize Audio2Face Connection & Settings (Optional, based on availability)
    print("Checking Audio2Face connection and applying initial settings...")
    if check_api_status():
        try:
            # Apply desired initial settings if A2F is running
            # load_usd_file() # Load default character if needed
            # activate_stream_livelink(activate=True) # Ensure livelink is active
            # set_stream_livelink_settings(settings={"enable_audio_stream": False}) # Disable audio over livelink
            set_audio_looping(loop=False) # Ensure audio doesn't loop by default
            print("✅ Audio2Face initial settings applied.")
        except Exception as e:
            # Log non-critical errors during optional setup
            print(f"⚠️ Warning: Failed to apply some initial Audio2Face settings: {e}")
            log_error(f"Failed during optional A2F initial setup: {e}")
    else:
        print("⚠️ Audio2Face API not detected. Skipping A2F initialization.")

    # 8. Start Continuous Emotion Detection Thread
    print("Starting continuous emotion detection thread...")
    emotion_thread = Thread(target=continuous_emotion_detection, daemon=True, name="EmotionDetector")
    emotion_thread.start()
    # Give it a moment to start up
    time.sleep(1)
    if emotion_thread.is_alive():
         print("✅ Continuous emotion detection thread appears active.")
    else:
         print("⚠️ Continuous emotion detection thread did not start correctly.")
         log_error("Emotion detection thread failed to start.")


    # 9. Initialize Response Counter
    response_counter = 1 # Start counter at 1
    print(f"✅ Response counter initialized to {response_counter}.")

    print("="*30 + " Initialization Complete " + "="*30 + "\n")


# --- Core Processing Logic ---
def process_user_input(user_question):
    """Handles user input, determines intent, interacts with services/LLM, and prepares response."""
    global response_counter, user_prefs, memory, conversation, current_emotion

    if not user_question:
        log_error("Received empty user input.")
        return "Sorry, I didn't get that. Could you please repeat?"

    print(f"\n--- Processing Input #{response_counter} ---")
    print(f"User Input: '{user_question}'")

    # 1. Get current detected emotion (thread-safe)
    with emotion_lock:
        user_emotion_detected = current_emotion
    print(f"Detected Emotion: {user_emotion_detected}")

    # 2. Analyze text sentiment
    user_text_sentiment = analyze_text_sentiment(user_question)
    print(f"Text Sentiment: {user_text_sentiment}")

    # Combine or prioritize emotion source (e.g., prefer facial if not neutral)
    effective_user_emotion = user_emotion_detected if user_emotion_detected != "neutral" else user_text_sentiment


    response_text = "I'm not sure how to respond to that yet." # Default fallback response
    intent_handled = False
    play_music_triggered = False # Flag to track if music should be played post-download

    # 3. Intent Detection & Handling
    user_question_lower = user_question.lower()

    # --- Weather Intent ---
    if re.search(r'\b(weather|temperature|forecast)\b', user_question_lower):
        print("Intent: Weather")
        try:
            # Extract city if mentioned, otherwise use preference or IP location
            city_match = re.search(r'(?:weather|temperature|forecast) in (\w+)', user_question_lower)
            city_to_check = city_match.group(1) if city_match else user_prefs.get("preferred_city")

            if not city_to_check: # If still no city, try IP based
                 location = get_location()
                 city_to_check = location.get("city", "London") # Fallback city
                 country = location.get("country", "")
                 print(f"City not specified, using location: {city_to_check}, {country}")

            weather_info = get_weather(city_to_check)
            if "Unable to fetch" in weather_info or "couldn't find" in weather_info :
                 response_text = f"Sorry, I had trouble getting the weather for {city_to_check}. Maybe try another city?"
            else:
                 response_text = f"The current weather in {city_to_check} is: {weather_info}."
            intent_handled = True
        except Exception as e:
            response_text = "Sorry, I encountered an error while checking the weather."
            print(f"❌ Error in Weather Intent: {e}")
            log_error(f"Weather intent error: {e}")

    # --- Music Intent ---
    elif re.search(r'\b(play|listen to|put on|start)\b.*\b(music|song|track|tune|artist|album)\b', user_question_lower) or \
         user_question_lower.startswith("play "):
        print("Intent: Music")
        song_name = ""
        # Try to extract song/artist name (this needs refinement for better NLP)
        match = re.search(r'(?:play|listen to|put on|start)\s+(?:(?:the|a|some)\s+)?(.*?)(?:\s+by\s+(.*))?$', user_question_lower.replace("music", "").replace("song", "").replace("track","").replace("tune",""))
        if match:
            song_name = match.group(1).strip()
            artist = match.group(2)
            if artist: song_name += f" by {artist.strip()}" # Append artist if found

        # Fallback if regex fails but intent was matched
        if not song_name and (user_question_lower.startswith("play ") or "play the song" in user_question_lower):
             song_name = re.sub(r'play(?: the song)?\s+', '', user_question, flags=re.IGNORECASE).strip()

        print(f"Extracted song query: '{song_name}'")

        # Handle cases where no specific song is mentioned
        if not song_name or song_name in ["music", "a song", "some music", ""]:
            # Suggest based on emotion or preference
            if effective_user_emotion in ["sad", "angry", "negative"]:
                song_name = f"calm {user_prefs.get('favorite_music_genre', 'instrumental')} music"
                response_text = f"Okay, you seem a bit {effective_user_emotion}. How about some calm {user_prefs.get('favorite_music_genre', 'instrumental')} music? Starting the download..."
            elif effective_user_emotion in ["happy", "positive"]:
                song_name = f"upbeat {user_prefs.get('favorite_music_genre', 'pop')} songs"
                response_text = f"Feeling {effective_user_emotion}! 😄 Let's get some upbeat {user_prefs.get('favorite_music_genre', 'pop')} music downloading!"
            else: # Neutral or preferred genre
                fav_genre = user_prefs.get("favorite_music_genre")
                if fav_genre:
                     song_name = f"{fav_genre} music"
                     response_text = f"Alright, downloading some {fav_genre} music as requested!"
                else:
                     song_name = "popular hits" # Generic fallback
                     response_text = "Okay, I'll look for some popular hits to download."
        else:
            # Confirmation for specific song
            response_text = f"Got it! Starting the download for '{song_name}'. I'll let you know when it's ready to play."

        # --- Download and Play Logic (using threads) ---
        if song_name:
            try:
                # Define the task for the download/play thread
                def download_and_play_task(s_name):
                    global response_counter # Need access to potentially update response later (tricky)
                    try:
                         print(f"Background Task: Starting download for '{s_name}'...")
                         downloaded_path = download_music(s_name) # This blocks this thread, not main Flask thread
                         if downloaded_path and os.path.exists(downloaded_path):
                              print(f"Background Task: Download complete for '{s_name}'. Path: {downloaded_path}")
                              print(f"Background Task: Initiating playback for {os.path.basename(downloaded_path)}...")
                              play_music_threaded(downloaded_path) # Start playback in yet another thread
                              # !! Cannot reliably send a *new* response ("Now playing...") back to the frontend from here !!
                              # This is a limitation of simple background threads in Flask.
                              # Solutions involve WebSockets, Server-Sent Events, or polling from frontend.
                         else:
                              print(f"❌ Background Task: Download failed or file missing for '{s_name}'. Cannot play.")
                              log_error(f"Background download/play task failed for {s_name}: Download unsuccessful.")
                              # Cannot easily inform user here either.
                    except Exception as e:
                         print(f"❌ Background Task: Error in download/play thread for '{s_name}': {e}")
                         log_error(f"Background download/play task error for {s_name}: {e}")

                # Start the download/play task in a background thread
                download_thread = Thread(target=download_and_play_task, args=(song_name,), daemon=True, name=f"Downloader_{song_name[:20]}")
                download_thread.start()
                intent_handled = True # Intent is handled by starting the download process

            except Exception as e:
                # Error initiating the download thread itself
                response_text = f"Sorry, there was an error starting the music download process for '{song_name}'. Please try again."
                print(f"❌ Error starting music download thread for '{song_name}': {e}")
                log_error(f"Error starting music download thread for '{song_name}': {e}")
        else:
             response_text = "I couldn't figure out what music you wanted. Can you be more specific?"
             intent_handled = True # Handled by responding


    # --- Email Intent --- (Requires more robust parsing in real use)
    elif re.search(r'\b(send|email|mail)\b', user_question_lower):
         print("Intent: Email")
         try:
             # VERY basic extraction - NEEDS proper NLP (e.g., spaCy, Rasa)
             to_match = re.search(r'to ([\w\.-]+@[\w\.-]+)', user_question_lower)
             subject_match = re.search(r'subject "?([\w\s]+)"?', user_question_lower)
             body_match = re.search(r'(?:saying|body|content) "?(.*)"?', user_question_lower) # Simplistic

             to_email = to_match.group(1) if to_match else None
             subject = subject_match.group(1).strip() if subject_match else "Update from Assistant"
             body_content = body_match.group(1).strip() if body_match else f"Regarding your request: '{user_question}'"

             if to_email:
                 # Ask for confirmation before sending (can't wait for response in HTTP)
                 # Instead, confirm *intent* and proceed, or state confirmation needed.
                 response_text = f"Okay, I can try to send an email to {to_email} with subject '{subject}'. Is that correct? (Note: I'll proceed based on this, please correct me if wrong)."
                 # In a real app, you'd likely store this intent and wait for a 'yes'/'confirm' message.
                 # For this example, we'll just send it. Add confirmation logic if needed.
                 # confirmation_needed = True
                 # if confirmation_needed:
                 #      response_text = f"Please confirm: Send email to {to_email} about '{subject}'?"
                 # else:
                 email_result = send_email_with_attachments(to_email, subject, body_content, []) # No attachments here
                 response_text = email_result # Return the result from send_email function
                 intent_handled = True

             else:
                 response_text = "Who should I send the email to? Please provide an email address."
                 intent_handled = True

         except Exception as e:
             response_text = "Sorry, I encountered an error while preparing the email."
             print(f"❌ Error in Email Intent: {e}")
             log_error(f"Email intent error: {e}")

    # --- News Intent ---
    elif re.search(r'\b(news|latest|updates|happening)\b', user_question_lower):
        print("Intent: News")
        query = ""
        # Extract query topic (simple extraction)
        match = re.search(r'(?:news|latest|updates) (?:on|about)\s+(.+)', user_question_lower, re.IGNORECASE)
        if match:
            query = match.group(1).strip()
        else: # Try to get keywords after the trigger word
             query = re.sub(r'.*\b(?:news|latest|updates|happening)\b\s*', '', user_question_lower, flags=re.IGNORECASE).strip()

        print(f"Extracted news query: '{query}'")

        if not query or query in ["on", "about", "this", ""]:
            # Suggest topic based on emotion or interests
            if effective_user_emotion in ["sad", "angry", "negative"]:
                 response_text = f"Hmm, maybe the news isn't the best idea if you're feeling {effective_user_emotion}. How about we talk about something else? Or I could find some positive news?"
            elif effective_user_emotion in ["happy", "positive"]:
                 query = random.choice(["good news", "inspiring stories", "tech highlights"] + user_prefs.get("interests",[]))
                 response_text = f"You seem {effective_user_emotion}! Let's find some interesting news. Checking for '{query}'..."
            else:
                 interests = user_prefs.get("interests", [])
                 if interests:
                     query = random.choice(interests)
                     response_text = f"What news are you interested in? How about '{query}' from your interests?"
                 else:
                     query = "world events" # Default fallback
                     response_text = "What news topic? I can check general world events."

        # Fetch news if a query was determined
        if query:
             try:
                 response_text = get_news(query)
                 intent_handled = True
             except Exception as e:
                 response_text = f"Sorry, I had trouble fetching the news about '{query}'."
                 print(f"❌ Error in News Intent fetch: {e}")
                 log_error(f"News intent fetch error for '{query}': {e}")
        # else: response_text was set in the suggestion block above
        intent_handled = True # Intent was handled even if asking for clarification


    # --- LLM Fallback ---
    if not intent_handled:
        print("Intent: General Conversation (LLM Fallback)")
        try:
            # Inject current emotion into context for the LLM (via prompt formatting)
            # Note: This requires the system prompt to have a placeholder like {user_emotion}
            # system_prompt_formatted = system_prompt.format(user_emotion=effective_user_emotion) # This needs adjusting if prompt is static

            # We'll rely on the LLM's general instruction to be adaptive for now.

            # Prepare input for the LLM chain
            current_chat_history = memory.load_memory_variables({})["chat_history"]
            inputs = {
                "human_input": user_question,
                "chat_history": current_chat_history,
                # "user_emotion": effective_user_emotion # Can add if prompt template supports it
            }

            # Invoke the LLM
            print("Invoking LLM...")
            llm_response = conversation.invoke(inputs)
            response_text = llm_response.content.strip()
            print(f"LLM Response: '{response_text[:100]}...'")


            # Update conversation memory with the turn
            # Important: Save context AFTER getting the response
            memory.save_context({"human_input": user_question}, {"output": response_text})
            print("Memory updated.")

        except Exception as e:
            response_text = "Sorry, I'm having a bit of trouble thinking right now. Could you try rephrasing?"
            print(f"❌ Error during LLM conversation: {e}")
            log_error(f"LLM conversation error: {e}")
            # Optionally clear last interaction from memory if it failed badly
            # try: memory.chat_memory.messages.pop() # Remove last AI attempt
            # except IndexError: pass


    # 4. Post-processing (TTS, A2F Emotion, A2F Audio)
    print("\n--- Post-processing Response ---")
    if response_text:
        try:
            # a) Analyze chatbot response sentiment
            chatbot_sentiment = analyze_text_sentiment(response_text)
            print(f"Chatbot Sentiment: {chatbot_sentiment}")

            # b) Generate A2F emotion payload (using effective user emotion and bot sentiment)
            print(f"Generating A2F payload based on User: {effective_user_emotion}, Bot: {chatbot_sentiment}")
            emotion_payload = generate_emotion_payload(effective_user_emotion, chatbot_sentiment)

            # c) Send emotion payload to A2F (non-blocking if possible)
            try:
                # Send in a separate thread to avoid blocking response generation?
                # Or use short timeout. For now, use short timeout.
                requests.post(f"{A2F_BASE_URL}/A2F/A2E/SetEmotionByName", json=emotion_payload, timeout=0.5) # Very short timeout
                print(f"Sent emotion payload to A2F: {emotion_payload['emotions']}")
            except requests.exceptions.Timeout:
                 print("⚠️ Timeout sending emotion payload to A2F (non-critical).")
                 # Log less frequently for timeouts if A2F might be slow/off
                 # log_error("Timeout sending emotion payload to A2F")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Warning: Failed to send emotion payload to A2F: {e}")
                # Log only if not a connection error (A2F likely off)
                if not isinstance(e, requests.exceptions.ConnectionError):
                     log_error(f"A2F emotion sending error: {e}")


            # d) Generate TTS audio for the response text
            file_name = f"response_{response_counter}.wav"
            print(f"Generating TTS for response #{response_counter}...")
            file_path = save_response_to_wav(response_text, file_name)

            # e) Send generated audio to A2F for playback/animation
            if file_path:
                print(f"Sending audio '{file_name}' to A2F...")
                a2f_send_result = send_audio_to_audio2face(file_name, response_counter)
                print(f"A2F audio send result: {a2f_send_result.get('status')}")
            else:
                print("⚠️ Warning: Skipping A2F audio send because TTS failed.")
                log_error(f"Skipped A2F audio send for response {response_counter} due to TTS failure.")

        except Exception as e:
            print(f"❌ Error during post-processing (Emotion/TTS/A2F): {e}")
            log_error(f"Post-processing error for response {response_counter}: {e}")
            # The response_text itself should still be valid unless TTS was the only source of error

    # 5. Increment response counter
    response_counter += 1

    # 6. Periodic Chat History Saving
    if (response_counter -1) % 5 == 0: # Save every 5 interactions
         print("Saving chat history periodically...")
         try:
             chat_history_list = [msg.dict() for msg in memory.load_memory_variables({})["chat_history"]]
             save_chat_history(chat_history_list)
             print("Chat history saved.")
         except Exception as e:
              print(f"⚠️ Warning: Failed to save chat history periodically: {e}")
              log_error(f"Failed to save chat history periodically: {e}")

    print(f"--- Responding to User --- \nResponse: '{response_text}'")
    print("-" * (60))
    return response_text

# --- Flask App Setup ---
app = Flask(__name__,
            template_folder='.', # Serve templates from the current directory
            static_folder='.')   # Serve static files (like CSS, JS if any) from current directory

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML frontend."""
    print("Serving index page (frontend.html)")
    # Ensure frontend.html is in the same directory as this script
    return render_template('frontend.html')

@app.route('/welcome_pa', methods=['GET'])
def welcome():
    """Provides the initial welcome message."""
    global response_counter # Ensure global counter is used
    print("Received request for welcome message.")
    welcome_message = "Hello there! I'm Meta, your personal assistant. How can I brighten your day?"

    # Generate audio and A2F for the welcome message
    try:
        file_name = f"response_{response_counter}.wav" # Use current counter
        file_path = save_response_to_wav(welcome_message, file_name)
        if file_path:
            # Set a default 'welcome' emotion (e.g., slightly positive/neutral)
            welcome_emotion = generate_emotion_payload("Neutral", "Positive")
            try:
                requests.post(f"{A2F_BASE_URL}/A2F/A2E/SetEmotionByName", json=welcome_emotion, timeout=0.5)
            except Exception as e:
                print(f"⚠️ Warning: Could not set welcome emotion for A2F: {e}")
                # Log only if not connection error
                if not isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                    log_error(f"A2F welcome emotion set error: {e}")

            # Send audio to A2F
            send_audio_to_audio2face(file_name, response_counter)
            response_counter += 1 # Increment counter after successful welcome message processing
        else:
             print("⚠️ Failed to generate welcome audio.")
             log_error("Failed to generate TTS for welcome message.")
    except Exception as e:
        print(f"❌ Error during welcome message audio/A2F generation: {e}")
        log_error(f"Welcome message post-processing error: {e}")

    return jsonify({"response": welcome_message})

@app.route('/chat_pa', methods=['POST'])
def chat():
    """Handles chat requests from the frontend."""
    start_time = time.time()
    print("\nReceived request on /chat_pa endpoint.")
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            print("❌ Invalid request: Missing 'user_input' in JSON payload.")
            log_error("Received invalid chat request (missing user_input).")
            return jsonify({"error": "No input received or invalid format"}), 400

        user_input = data['user_input']
        print(f"User input from frontend: '{user_input}'")

        # Process the input using the main logic function
        bot_response = process_user_input(user_input)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Request processed in {processing_time:.2f} seconds.")

        return jsonify({"response": bot_response})

    except Exception as e:
        # Catch-all for unexpected errors during request processing
        print(f"❌ Critical Error in /chat_pa endpoint: {e}")
        # Include traceback for debugging if possible/needed (be careful in production)
        import traceback
        tb_str = traceback.format_exc()
        print(tb_str)
        log_error(f"Critical Error in /chat_pa: {e}\nTraceback:\n{tb_str}")
        return jsonify({"error": "An unexpected internal server error occurred. Please try again later."}), 500


# --- Main Execution ---
if __name__ == "__main__":
    initialize_app() # Run all initializations first

    print("\n🚀 Starting Flask Server...")
    print("   Access the frontend at: http://127.0.0.1:5001 (or your local IP)")
    print("   Ensure Audio2Face and News Service (if used) are running.")
    print("   Press CTRL+C to stop the server.")

    # Run Flask server
    # Use threaded=True for the development server to handle multiple requests
    # and allow background threads (like emotion detection, music) to run.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    # use_reloader=False is important to prevent initialize_app() running twice in debug mode.
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n⏹️ Server shutdown requested (KeyboardInterrupt).")
    except Exception as e:
         print(f"\n🚨 Flask server failed to start or crashed: {e}")
         log_error(f"Flask server run error: {e}")
    finally:
        # --- Cleanup code (runs on graceful shutdown/exit) ---
        print("\n--- Initiating Server Shutdown & Cleanup ---")
        emotion_tracking_active = False # Signal emotion thread to stop
        print("Signaled emotion tracking thread to stop.")

        # Attempt to stop any ongoing A2F animation
        print("Attempting to stop A2F animation...")
        stop_audio2face_animation()

        # Save final chat history
        print("Saving final chat history...")
        if memory:
            try:
                 chat_history_list = [msg.dict() for msg in memory.load_memory_variables({})["chat_history"]]
                 save_chat_history(chat_history_list)
                 print("✅ Final chat history saved.")
            except Exception as e:
                 print(f"⚠️ Failed to save final chat history during shutdown: {e}")
                 log_error(f"Failed to save final chat history during shutdown: {e}")
        else:
             print("⚠️ Memory object not found, cannot save final history.")

        print("Cleanup complete. Goodbye!")
        # Allow threads a moment to potentially finish cleanup
        time.sleep(1)