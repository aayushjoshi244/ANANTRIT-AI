import os
import subprocess
from datetime import datetime

def speak(text: str) -> None:
    subprocess.run(["espeak", text], check=False)

def handle_command(command: str) -> bool:
    cmd = command.lower().strip()

    if "hello" in cmd:
        speak("Hello Aayush")
    elif "time" in cmd:
        now = datetime.now().strftime("%I %M %p")
        speak(f"The time is {now}")
    elif "date" in cmd:
        today = datetime.now().strftime("%d %B %Y")
        speak(f"Today's date is {today}")
    elif "open cctv" in cmd:
        speak("Opening CCTV stream")
        os.system("xdg-open http://127.0.0.1:5000")
    elif "start guard" in cmd:
        speak("Starting guard mode")
        os.system("python3 smart_guard.py &")
    elif "stop guard" in cmd:
        speak("Stopping guard mode")
        os.system("pkill -f smart_guard.py")
    elif "shutdown" in cmd:
        speak("Shutting down Raspberry Pi")
        os.system("sudo shutdown now")
    elif "reboot" in cmd:
        speak("Rebooting Raspberry Pi")
        os.system("sudo reboot")
    elif "exit" in cmd or "quit" in cmd:
        speak("Goodbye")
        return False
    else:
        speak("Sorry, I did not understand that")
    return True

def main():
    speak("Jarvis is ready")
    while True:
        command = input("Say command: ")
        if not handle_command(command):
            break

if __name__ == "__main__":
    main()