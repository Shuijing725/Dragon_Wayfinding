# NLU
import os

from rasa.core.agent import Agent # pip install rasa
import asyncio

# ROS
import rospy
from std_msgs.msg import String


import tensorflow as tf
import argparse


class RASA_NLU(object):
    def __init__(self, args):
        # NLU
        self.agent = Agent.load(model_path=args.nlu_model_path)
        self.text_out = None

    def output(self, message):
        message = message.strip()
        result = asyncio.run(self.agent.parse_message(message))
        return result



class NLUIntentParser():
    def __init__(self, args):
        self.nlu = RASA_NLU(args)


    def nlu_intent_parser(self, text_msg):
        sentence = text_msg.data
        print('sentence:', sentence)
        nlu_out = self.nlu.output(sentence)
        # print(nlu_out)
        intent = nlu_out['intent']['name']
        print('intent:', intent)
        if intent in ['say_goal_location', 'say_goal_object']:
            try:
                entity = nlu_out['entities'][0]['value']
                print(nlu_out['entities'])
            except IndexError:
                entity = None
        else:
            entity = None


    def run(self):

        # Unit test: run NLU if we detect a sentence spliter (period) from speech-to-text
        test_msg = String()
        test_msg.data = "hello robot"
        self.nlu_intent_parser(test_msg)

        test_msg = String()
        test_msg.data = "Take me to the entrance near the office"
        self.nlu_intent_parser(test_msg)

        # unable to distinguish target and reference object
        test_msg = String()
        test_msg.data = "A clear door near the sofa"
        self.nlu_intent_parser(test_msg)

        test_msg = String()
        test_msg.data = "A table and some chairs"
        self.nlu_intent_parser(test_msg)

        test_msg = String()
        test_msg.data = "Go to a vending machine"
        self.nlu_intent_parser(test_msg)

        test_msg = String()
        test_msg.data = "Take me to a computer desk"
        self.nlu_intent_parser(test_msg)

        print('goal sent')
        rospy.sleep(3)
        test_msg = String()
        test_msg.data = "Go to a table in the dining room"
        self.nlu_intent_parser(test_msg)

        rospy.sleep(3)
        test_msg = String()

        test_msg.data = "Please take me to somehwere to sit in the lab"
        self.nlu_intent_parser(test_msg)

        rospy.sleep(3)
        test_msg = String()
        test_msg.data = "Goodbye"
        self.nlu_intent_parser(test_msg)




if __name__ == "__main__":
    # to prevent tensorflow from taking all gpu memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--record_images', default=True, action='store_true')
    parser.add_argument('--nlu_model_path', type=str, default="pretrained_models/nlu.tar.gz")

    args = parser.parse_args()

    parser = NLUIntentParser(args)
    parser.run()
