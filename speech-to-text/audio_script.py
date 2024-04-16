#! /usr/bin/env python3

"""
Node listens to audio data being output to the 'audio' topic
and publishes the recognized text to the 'destination' topic.

If unable to recognize the audio, the node will output "".

NOTE: 
Make sure that `roslaunch audio_capture capture.launch` is run 
before this script as it listens to the output on the audio/audio
topic published by the audio_capture node.

run `python3 -m pip install git+https://github.com/openai/whisper.git`
before running the file the first time.
"""

import rospy
import numpy as np
import speech_recognition as sr
import  os
import io
import pdb
import datetime
from pydub import AudioSegment

from scipy.io.wavfile import write, read
from audio_common_msgs.msg import AudioData
from std_msgs.msg import String

store_captured_audio = True

if store_captured_audio:
    counter = 0
    save_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'user')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
else:
    # names of the temporary audio files
    TEMP_WAV_PATH = 'temp.wav'
TEMP_MP3_PATH = 'temp.mp3'

# frame size can be changed as a hyperparameter for longer or shorter recognition periods
FRAME_SIZE = 150
frame_num = 0
audio_data = []
ignore = 0

recognizer = sr.Recognizer()
publisher = None

def recognize(data):
    """
    Performs recognizion on the audio data input from the audio_capture node

    Inputs:
        data: AudioData: message with the audio data
    Returns: 
        None. Outputs to the Publisher
    """
    global frame_num, audio_data, publisher, recognizer, ignore, store_captured_audio


    audio_data.append(data.data)    # audio_data is a list of bytes
    frame_num += 1


    # delete previous temporary files
    if os.path.exists(TEMP_MP3_PATH):
        os.remove(TEMP_MP3_PATH)

        
    # Perform recognition over the last FRAME_SIZE frames
    if len(audio_data) % FRAME_SIZE == FRAME_SIZE -1:
        # writing the bytes to the mp3_file
        mp3_file = open(TEMP_MP3_PATH, "wb")
        mp3_file.write(b''.join(audio_data)) 
        mp3_file.close()
        
        # remove the oldest bit of audio
        audio_data = [] 

        # the mp3 file is then convereted to wav which can be used as input by the recognizer
        # this is a work around since the audio source for the recognizer could not be mp3

        # import the saved audio data in the right format    
        sound = AudioSegment.from_mp3(TEMP_MP3_PATH)

        if store_captured_audio:
            global save_dir, counter
            wav_path = os.path.join(save_dir, str(counter) + '.wav')
            counter = counter + 1
        else:
            if os.path.exists(TEMP_WAV_PATH):
                os.remove(TEMP_WAV_PATH)
            wav_path = TEMP_WAV_PATH
        sound.export(wav_path, format = "wav")
        
        try:
            with sr.AudioFile(wav_path) as source:
                adata = recognizer.record(source)
                recognized_text = recognizer.recognize_whisper(adata,language='english', model='small.en') # model='tiny.en'
                print('Human:', recognized_text)
                publisher.publish(recognized_text)
                ignore = 1 # this is currently meaningless 
        except sr.UnknownValueError:
            ignore = 0
            print("noise")

def main():
    global publisher

    rospy.init_node('recognizer', anonymous=True)
    
    # Create publishing topic
    publisher = rospy.Publisher('text', String, queue_size=10)
    print("Listening")

    # Subscribe to audio so that recognizer is called whenever there is audio data published
    rospy.Subscriber('/audio/audio', AudioData, recognize)
    rospy.spin()

if __name__ == '__main__':
    main()
