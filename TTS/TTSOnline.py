from gtts import gTTS
import os

def text_to_speech_gtts(text, lang='en', output_file='output.mp3'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)
        os.system(f"start {output_file}" if os.name == 'nt' else f"open {output_file}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
text = "hello Himanshu Singh, I am your new TTS assistant ."

text_to_speech_gtts(text)
