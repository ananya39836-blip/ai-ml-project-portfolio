from gtts import gTTS
import os

print("=" * 40)
print("TEXT TO SPEECH (NO TRANSLATION)")
print("=" * 40)

text = input("Enter text: ")

print("\nLanguage Codes:")
print("en = English")
print("hi = Hindi")
print("fr = French")
print("es = Spanish")
print("de = German")
print("ta = Tamil")
print("te = Telugu")
print("ja = Japanese")
print("ko = Korean")
print("ar = Arabic")

lang = input("\nEnter language code: ").strip()

try:
    tts = gTTS(text=text, lang=lang, slow=False)

    filename = f"speech_{lang}.mp3"
    tts.save(filename)

    print("\nAudio saved:", filename)

    # Open file automatically
    os.startfile(filename)

except Exception as e:
    print("\nError occurred:")
    print(e)
speed = input("Slow voice? (y/n): ").lower()

slow = True if speed=="y" else False

tts = gTTS(
    text=text,
    lang=lang,
    slow=slow
)
from datetime import datetime

filename = datetime.now().strftime(
    "speech_%Y%m%d_%H%M%S.mp3"
)

tts.save(filename)
choice = input(
"1. Type text\n2. Read from file\nChoose: "
)

if choice=="1":
    text=input("Enter text:\n")

elif choice=="2":
    file=input("Enter txt filename: ")

    with open(file,"r",encoding="utf8") as f:
        text=f.read()
words=len(text.split())
chars=len(text)

print("Words:",words)
print("Characters:",chars)

