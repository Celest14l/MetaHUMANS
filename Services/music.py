import os
import yt_dlp
import vlc
import time
import speech_recognition as sr
from threading import Thread
import sys
import ctypes

# ✅ VLC Installation Path
vlc_path = "C:\VLC"

# ✅ Add VLC to system path
os.environ["PATH"] += os.pathsep + vlc_path
os.environ["VLC_PLUGIN_PATH"] = os.path.join(vlc_path, "plugins")

# ✅ Load VLC DLL manually
dll_path = os.path.join(vlc_path, "libvlc.dll")

if os.path.exists(dll_path):
    try:
        ctypes.CDLL(dll_path)
        print("✅ VLC successfully loaded!")
    except Exception as e:
        print(f"❌ Error loading VLC: {e}")
        sys.exit(1)
else:
    print(f"❌ VLC library not found at {dll_path}! Check your VLC installation path.")
    sys.exit(1)

def recognize_speech():
    """Recognize speech command from the user"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎙️ Speak your command (Play, Pause, Skip)...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"🗣️ You said: {command}")
            return command
        except sr.WaitTimeoutError:
            print("⏳ No speech detected, try again!")
        except sr.UnknownValueError:
            print("❌ Couldn't understand, try again!")
        except sr.RequestError:
            print("🚨 Speech recognition service is down!")
        return None

def download_music(song_name):
    """Download music using yt-dlp"""
    print(f"🔍 Searching for: {song_name}")
    
    file_path = "music.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,  # Avoid downloading full playlists
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': file_path,  # Ensures correct filename extension
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"ytsearch:{song_name}"])

    return file_path

def play_music(file_path):
    """Play music using VLC with manual and speech controls"""
    player = vlc.MediaPlayer(file_path) # type: ignore
    player.play()

    print("\n🎵 Controls: Say 'Pause', 'Play', or 'Skip' OR Enter [0] Pause | [1] Resume | [2] Skip")
    
    while True:
        command = recognize_speech() or input("Enter command: ").strip().lower()

        if command in ["0", "pause"]:
            player.pause()
            print("⏸ Music Paused")

        elif command in ["1", "play", "resume"]:
            player.play()
            print("▶️ Music Resumed")

        elif command in ["2", "skip", "next"]:
            player.stop()
            print("⏭ Skipping to the next song...")
            if os.path.exists(file_path):
                os.remove(file_path)
            return  # Exit playback function

        time.sleep(1)

if __name__ == "__main__":
    while True:
        print("\n🎧 Choose Input Mode:")
        print("[1] Voice Input")
        print("[2] Text Input")
        print("[3] Exit")

        mode = input("\nEnter your choice (1/2/3): ").strip()

        if mode == "1":
            print("\n🎙️ Say 'Play' followed by a song name...")
            song_command = recognize_speech()
            
            if song_command and "play" in song_command:
                song_name = song_command.replace("play", "").strip()
            else:
                print("❌ Could not recognize song name. Try again.")
                continue
        
        elif mode == "2":
            song_name = input("\nEnter song name (or 'exit' to quit): ").strip()
            if song_name.lower() == "exit":
                print("👋 Exiting Music Player...")
                break
        
        elif mode == "3":
            print("👋 Exiting Music Player...")
            break
        
        else:
            print("⚠️ Invalid choice! Please enter 1, 2, or 3.")
            continue

        file_path = download_music(song_name)
        music_thread = Thread(target=play_music, args=(file_path,))
        music_thread.start()
        music_thread.join()
