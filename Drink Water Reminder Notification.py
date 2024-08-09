from gtts import gTTS
from plyer import notification
import os
import time
import threading

while(True):
    def text_to_speech(text, language='en'):
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save the audio file
        tts.save('Drink_Water_Reminder.mp3')
        
        # Play the audio file
        os.system("start Drink_Water_Reminder.mp3")  # For Windows
        # Use the following line for macOS:
        # os.system("open output.mp3")
        # Use the following line for Linux:
        # os.system("xdg-open output.mp3")

    def send_notification(title, message):
        notification.notify(
            title=title,
            message=message,
            app_name='Notification App',
            timeout=10  # Duration in seconds
        )
    
    if __name__ == "__main__":
        text = "please drink water"
        f1 = threading.Thread(target=text_to_speech, args=[text])
        f2 = threading.Thread(target=send_notification, args=["Reminder", "Please Drink Water!"])
        f1.start()
        f2.start()
    time.sleep(7200)         #Notification after 2 hours