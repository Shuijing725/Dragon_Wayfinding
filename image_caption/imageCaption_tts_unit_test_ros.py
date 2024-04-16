"""
Put this file in the remove tb2 jetson xaiver, and run ros_inference.py before running it
Dependencies:
- Go through the setup from https://codelabs.developers.google.com/codelabs/cloud-text-speech-python3#1
- pip3 install --user --upgrade google-cloud-texttospeech
- pip install playsound pygobject
Subscribe: /image_caption_text (published by ros_inference.py)
Publish: None
"""
import os
import rospy
from std_msgs.msg import String

import google.cloud.texttospeech as tts
from playsound import playsound
import time
import datetime
import re

class Speaker:

    def __init__(self):
        self.voice_name = 'en-US-Neural2-C'
        self.language_code = "-".join(self.voice_name.split("-")[:2])
        self.voice_params = tts.VoiceSelectionParams(
            language_code=self.language_code, name=self.voice_name
        )
        self.audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)
        self.client = tts.TextToSpeechClient()

        # save all audio in a folder
        self.save_dir = os.path.join(os.getcwd(), 'data', datetime.datetime.now().strftime('%m-%d_%H-%M-%S')+'-tts')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.counter = 0

    def text_to_speech_callback(self, text_msg):
        # convert text to mp3 file with google api
        text = text_msg.data
        text_input = tts.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            input=text_input, voice=self.voice_params, audio_config=self.audio_config)

        # remove spaces and special symbols in text, and save it as part of wav file name
        text_filename = text.replace(" ", "_")
        text_filename = re.sub('[^a-zA-Z0-9]+', '', text_filename)

        filename = os.path.join(self.save_dir, str(self.counter)+'_'+text_filename+".wav")
        self.counter = self.counter + 1
        with open(filename, "wb") as out:
            out.write(response.audio_content)
            print(f'Generated speech saved to "{filename}"')

        # send the sound data to the microphone on tb2
        playsound(filename)




if __name__ == '__main__':
    try:
        rospy.init_node('caption_speaker', log_level=rospy.INFO, disable_signals=True)
        rospy.loginfo('caption_speaker node started')

        speaker = Speaker()
        caption_sub = rospy.Subscriber('/image_caption_text', String, speaker.text_to_speech_callback)

        while not rospy.is_shutdown():
            rospy.spin()

    except rospy.ROSInterruptException:
        exit()