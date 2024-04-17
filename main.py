# NLU
from rasa.core.agent import Agent # pip install rasa
import asyncio

# ROS
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, CompressedImage, LaserScan
import dynamic_reconfigure.client

from cv_bridge import CvBridge
import cv2

# find goal
from semantic_map.set_goal_pose_clip import SelectGoalClip

# image captioning
from image_caption.image_caption_unit_test_ros import ImageCaptioner

# vqa
from VQA.vqa_unit_test_ros import VQA

import tensorflow as tf
import argparse
import datetime
import os
import numpy as np

"""
Given the parsed intent and entities (and the original sentence), call corresponding downstream modules to fulfill the intent
"""
class Executer(object):
    '''
    args: the arguments defined in main function below
    pub: the ROS publisher that publishes the robot feedback text (will be narrated to the user with text-to-speech)
    '''
    def __init__(self, args, pub):
        # NLU
        self.agent = Agent.load(model_path=args.nlu_model_path)
        self.text_out = None

        # ROS
        self.goal_selection_method = args.goal_selector
        # find goal clip
        self.goal_sender = SelectGoalClip(landmark_folder=args.landmark_folder,
                                          pub = pub,
                                          method=self.goal_selection_method,
                                          clip_model_preprocessor=None,
                                          custom_clip_model_path=args.clip_model_path)
        self.publisher = pub

        # image captioning
        self.image_captioner = ImageCaptioner(pub=self.publisher)
        self.image_topic_name = args.image_topic_name

        # vqa
        self.vqa = VQA(model_path=args.vqa_model_path, pub=self.publisher)

        # flag: 1 if a goal is being executed, 0 otherwise
        self.exe_goal_flag = 0
        # flag: 1 if the robot is in a pause (it will resume later), 0 otherwise
        self.pause_flag = 0

        # flag: [attribute is filled or no, object is filled or no]
        self.goal_confirm_flag = [0, 0]
        # variable to store unconfirmed goal sentence and entity
        self.goal_sentence = None
        self.goal_entity = ''

        # location of objects, for disambiguation
        self.goal_location = None

        # if True, ignore all text (to avoid random robot behaviors when user is talking to someone else)
        # if False, the program run as usual
        # self.sleep is set to True by intent 'greet', set to False by intent 'sleep'
        self.sleep = False

        # for changing the max speed
        # for changing the max speed
        self.vel_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS", timeout=2)
        # print(self.vel_client.get_configuration(timeout=2))
        all_params = self.vel_client.get_configuration(timeout=2)
        self.cur_trans_vel = all_params['max_vel_trans']
        self.max_trans_vel = 0.5
        self.min_trans_vel = 0.15

        self.cur_rot_vel = all_params['max_vel_theta']
        self.max_rot_vel = 5.0
        self.min_rot_vel = 1.0


        # Use a long greet text in the first greet, otherwise us a shorter text
        self.greet_text_long = "Hey!, What can I do for you?"
        self.greet_text_short = "Hey!, What can I do for you?"
        self.first_greet = True

    def output(self, message):
        message = message.strip()
        result = asyncio.run(self.agent.parse_message(message))
        return result

    # Use an object detector to describe the scene
    def describe_the_scene(self):
        data = rospy.wait_for_message(self.image_topic_name, CompressedImage, timeout=5)
        self.image_captioner.image_captioning_callback(data)

    # Pause the robot
    def pause_robot(self):
        self.goal_sender.pause_robot()
        self.pause_flag = 1
        txt_response = 'Sure, taking a pause now'
        return txt_response

    # Resume the robot
    def resume_robot(self):
        self.goal_sender.resume_robot()
        self.pause_flag = 0
        txt_response = 'Resuming to the original destination'
        return txt_response

    # Change the speed of the robot
    def change_speed(self, faster):
        """
        faster = True: speed up
        faster = False: slow down
        """
        # https://answers.ros.org/question/359120/modifying-parameter-values-rqt_reconfigure-using-a-python-script/
        # check which params to change
        if faster:
            print('speed up')
            new_trans_vel = self.cur_trans_vel + 0.1
            new_rot_vel = self.cur_rot_vel + 1.
        else:
            print('speed down')
            new_trans_vel = self.cur_trans_vel - 0.1
            new_rot_vel = self.cur_rot_vel - 1.
        self.cur_trans_vel = np.clip(new_trans_vel, self.min_trans_vel, self.max_trans_vel)
        self.cur_rot_vel = np.clip(new_rot_vel, self.min_rot_vel, self.max_rot_vel)

        print('new v:', self.cur_trans_vel, 'new w:', self.cur_rot_vel)
        self.vel_client.update_configuration({'max_vel_trans': self.cur_trans_vel, 'max_vel_theta': self.cur_rot_vel})

    # Call the landmark recognizer (CLIP or detector) to send the goal pose to robot
    def set_goal(self, text):
        # print('begin to send goal')
        ret_val = self.goal_sender.send_goal(text, wait=False)
        self.exe_goal_flag = 1
        # print('goal sent')
        return ret_val

    # Answer visual questions about the environment
    def answer_visual_questions(self):
        data = rospy.wait_for_message(self.image_topic_name, CompressedImage, timeout=5)
        self.vqa.vqa_callback(data)

    # given the parsed intent and other information, call the corresponding downstream functions and generate robot language feedback
    def parse_intent(self, intent, sentence, entity=None, attribute=None):
        if intent == 'sleep':
            self.sleep = True

        if self.sleep:
            if intent == 'greet':
                self.sleep = False
                if self.first_greet:
                    self.text_out = self.greet_text_long
                    self.first_greet = False
                else:
                    self.text_out = self.greet_text_short
                # publish text as String, for tb2 microphone to vocalize
                print(self.text_out)
                self.publisher.publish(self.text_out)
            else:
                return
        else:
            if intent=='greet':
                if self.first_greet:
                    self.text_out = self.greet_text_long
                    self.first_greet = False
                else:
                    self.text_out = self.greet_text_short
                # publish text as String, for tb2 microphone to vocalize
                print(self.text_out)
                self.publisher.publish(self.text_out)

            elif intent=='goodbye':
                self.text_out = "Bye"
                print(self.text_out)
                self.publisher.publish(self.text_out)

            elif intent in ['affirm', 'say_goal_object+affirm'] and self.goal_confirm_flag == [1, 1]:
                self.pause_flag = 0
                # (a dining chair) + (in the kitchen)
                if self.goal_location is not None or len(self.goal_entity) > 0:
                    # if clip: concat the goal_entity and goal_location
                    if self.goal_selection_method == 'clip':
                        if self.goal_location is None:
                            self.goal_location = ''
                        if len(self.goal_entity) == 0:
                            self.goal_entity = 'somewhere'
                        # (a dining chair) + (in the kitchen)
                        self.goal_entity = self.goal_entity + ' ' + self.goal_location


                    goal_sent = self.set_goal(self.goal_entity)
                    if goal_sent == 0:
                        self.text_out = 'Sure, taking you to ' + self.goal_entity
                    else:
                        self.text_out = ''
                else:
                    goal_sent = self.set_goal(self.goal_sentence)
                    if goal_sent == 0:
                        self.text_out = 'Sure, taking you to your destination'
                    else:
                        self.text_out = ''
                # reset the variables after sending the goal to mobile base
                self.goal_confirm_flag = [0, 0]
                self.goal_sentence = None
                self.goal_entity = ''
                self.goal_location = None
                print(self.text_out)
                self.publisher.publish(self.text_out)

            elif intent == 'deny' and self.goal_confirm_flag == [1, 1]:
                # go back to unconfirmed state
                self.goal_confirm_flag = [0, 0]
                self.goal_sentence = None
                self.goal_entity = ''
                self.goal_location = None
                self.text_out = "Can you provide some details about your destination?"
                print(self.text_out)
                self.publisher.publish(self.text_out)

            elif intent == 'describe_the_scene':
                self.describe_the_scene()

            elif intent in ['say_goal_object', 'say_goal_object+say_goal_location', 'say_goal_object+deny',
                            'say_goal_object+greet', 'say_goal_object+say_goal_location+greet']:
                if self.goal_selection_method == 'clip':
                    # the user sentence is clear, no need for disambiguation
                    if entity and attribute:
                        # 1. update variables
                        self.goal_entity = 'a ' + attribute + ' ' + entity
                        # 2. change confirm flag
                        new_goal_confirm_flag = [1, 1]
                        # 3. create output sentence
                        self.text_out = 'Do you wish to go to ' + self.goal_entity
                    elif entity and not attribute:
                        if self.goal_confirm_flag == [1, 0]: # attribute is filled, object is not filled
                            self.goal_entity = self.goal_entity + ' ' + entity
                            new_goal_confirm_flag = [1, 1]
                            self.text_out = 'Do you wish to go to ' + self.goal_entity
                        # [0, 0]: if we don't have any memory of attribute or obj
                        # [0, 1]/[1, 1]: if the user said an object (probably with attr) before, but they change mind and said a new object now
                        else:
                            self.goal_entity = entity
                            # the user said an ambiguious table or chair
                            if entity in ['table', 'chair']:
                                new_goal_confirm_flag = [0, 1] # attribute is not filled, object is filled
                                if entity == 'chair':
                                    self.text_out = 'What kind of chair are you looking for? For example, a dining chair, an office chair, or a sofa?'
                                else:
                                    self.text_out = 'What kind of table are you looking for? For example, a dining table or an office desk?'
                            # the user said an object that does not need disambiguation (any object except table and chair)
                            else:
                                new_goal_confirm_flag = [1, 1]
                                self.text_out = 'Do you wish to go to ' + self.goal_entity


                    elif attribute and not entity:
                        # set or update the attribute
                        if self.goal_confirm_flag == [0, 0] or self.goal_confirm_flag == [1, 0]:
                            # the user said an attribute, but did not say an object
                            self.goal_entity = 'a ' + attribute
                            new_goal_confirm_flag = [1, 0]
                            self.text_out = 'What object are you looking for?'
                        elif self.goal_confirm_flag == [0, 1]: # missing an attribute before
                            self.goal_entity = 'a ' + attribute + ' ' + self.goal_entity
                            new_goal_confirm_flag = [1, 1]
                            self.text_out = 'Do you wish to go to ' + self.goal_entity
                        # the user said an object (probably with attr) before, but said a new attr now
                        else:
                            # remove 'a ', add the new attr
                            self.goal_entity = 'a ' + attribute + ' ' + self.goal_entity[2:]
                            new_goal_confirm_flag = [1, 1]
                            self.text_out = 'Do you wish to go to ' + self.goal_entity
                    else:
                        # no entity or attribute extracted from NLU
                        self.goal_sentence = sentence
                        new_goal_confirm_flag = [1, 1]
                        self.text_out = 'Do you wish to go to ' + self.goal_sentence
                # if we use an object detector to find goals, disambiguation is not possible
                else:
                    new_goal_confirm_flag = [1, 1]
                    self.goal_sentence = sentence
                    if entity:
                        self.goal_entity = entity
                        self.text_out = 'Do you wish to go to ' + entity
                    else:
                        self.text_out = 'Do you wish to go to ' + sentence

                print(self.text_out)
                self.publisher.publish(self.text_out)
                print('self.goal_confirm_flag before update:', self.goal_confirm_flag)
                self.goal_confirm_flag = new_goal_confirm_flag
                print('self.goal_confirm_flag after update:', self.goal_confirm_flag)

            elif intent in ['say_goal_location', 'say_goal_location+greet']:
                if attribute:
                    self.text_out = "Please describe what object you are looking for in the " + attribute
                    self.goal_location = 'in the ' + attribute
                    self.goal_confirm_flag = [1, 0]
                else:
                    self.text_out = "Can you provide some details about your destination?"
                print(self.text_out)
                self.publisher.publish(self.text_out)

            # the robot can only be paused or resumed when it is executing some goal
            elif intent == 'pause':
                if self.exe_goal_flag == 1:
                    self.text_out = self.pause_robot()
                    print(self.text_out)
                    self.publisher.publish(self.text_out)

            elif intent == 'resume':
                if self.exe_goal_flag == 1:
                    self.text_out = self.resume_robot()
                    print(self.text_out)
                    self.publisher.publish(self.text_out)

            elif intent == 'accelerate':
                self.change_speed(faster=True)
                self.text_out = 'Sure, increase my speed from now'
                print(self.text_out)
                self.publisher.publish(self.text_out)

            elif intent == 'decelerate':
                self.change_speed(faster=False)
                self.text_out = 'Sure, decrease my speed from now'
                print(self.text_out)
                self.publisher.publish(self.text_out)

            elif intent == 'ask_question':
                self.vqa.set_question(sentence)
                self.answer_visual_questions()

            elif intent == 'unknown':
                # self.text_out = "Sorry, I didn't get that. Can you say it again?"
                # self.publisher.publish(self.text_out)
                pass
            else:
              pass

    # when the robot arrives at a goal, publish a goal arrival message to inform the user
    def announce_goal_arrival(self, data):
        if self.exe_goal_flag == 1:
            wait = self.goal_sender.client.wait_for_result()
            if not wait:
                rospy.logerr("Action server not available!")
                rospy.signal_shutdown("Action server not available!")
            else:
                if self.pause_flag == 0:
                    self.publisher.publish('Goal arrived!')
                    self.exe_goal_flag = 0

'''
Given a text command from speech recognition, extract the intent and entities with rasa NLU model, and call corresponding downstream modules
'''
class NLUIntentParser():
    def __init__(self, args):
        rospy.init_node("main")
        # publishers
        # publisher for the caption text
        text_pub = rospy.Publisher('/image_caption_text', String, queue_size=2)

        self.nlu = Executer(args, pub=text_pub)

        # subscribers
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.nlu.announce_goal_arrival)
        rospy.Subscriber("/scan", LaserScan, self.nlu.goal_sender.update_goal)

        if args.record_images:
            print('begin recording')
            rospy.Subscriber(args.image_topic_name, CompressedImage, self.image_saver)
            self.bridge = CvBridge()
            self.img_counter = 0
            self.save_dir = os.path.join(os.getcwd(), 'image_temp', datetime.datetime.now().strftime('%m-%d_%H-%M-%S')+'-images')
            os.makedirs(self.save_dir)

    def image_saver(self, img_msg):
        if self.img_counter % 5 == 0:
            # print('record', self.img_counter)
            color_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            # save the image for debugging
            filename = str(int(self.img_counter//5)) + ".png"
            cv2.imwrite(os.path.join(self.save_dir, filename), color_image)
        self.img_counter = self.img_counter + 1

    # convert text to all lower cases, and remove heading and tailing spaces
    def process_text(self, text):
        if text[0] == ' ':
            text = text[1:]
        if text[-1] == ' ':
            text = text[:-1]
        return text.lower()

    # input: text message from speech recognition
    # output: the intents and entities parsed from the text
    def nlu_intent_parser(self, text_msg):
        sentence = text_msg.data
        nlu_out = self.nlu.output(sentence)
        # print(nlu_out)
        intent = nlu_out['intent']['name']

        if intent in ['say_goal_location', 'say_goal_object', 'say_goal_object+say_goal_location', 'say_goal_object+deny',
                      'say_goal_object+affirm', 'say_goal_object+greet', 'say_goal_location+greet', 'say_goal_object+say_goal_location+greet']:
            # entity: object, attribute: everything else (see entities in domain.yml)
            try:
                entity = None
                attribute = None
                for i in range(len(nlu_out['entities'])):
                    # print(nlu_out['entities'][i]['entity'], 'with confidence:', nlu_out['entities'][i]['confidence'])
                    if nlu_out['entities'][i]['entity'] == 'verb':
                        attribute = self.process_text(nlu_out['entities'][i]['value']) + 'ing'
                    elif nlu_out['entities'][i]['entity'] in ['adjective', 'office', 'kitchen', 'lounge']:
                        attribute = self.process_text(nlu_out['entities'][i]['value'])
                    else:
                        entity = self.process_text(nlu_out['entities'][i]['value'])
                # print(attribute)
            except IndexError:
                entity = None
                attribute = None
                print('No attribute found')
        else:
            entity = None
            attribute = None
        print('\nintent:', intent, ', sentence:', sentence, ', entity:', entity, ', attribute:', attribute)
        self.nlu.parse_intent(intent, sentence, entity, attribute)

    # if a /text message from speech recognition comes, parse the text
    def run(self):
        # todo: Whole test: subscribe to Aamir's code
        while not rospy.is_shutdown():
            data = rospy.wait_for_message('/text', String, timeout=1000)
            print('Human:', data)
            self.nlu_intent_parser(data)



if __name__ == "__main__":
    # to prevent tensorflow from taking all gpu memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser('Parse configuration file')
    # whether you want to record all camera images or not (Warning: the recorded image will take lots of disk space!)
    parser.add_argument('--record_images', default=False, action='store_true')
    # the path of trained NLU model
    parser.add_argument('--nlu_model_path', type=str, default="pretrained_models/nlu.tar.gz")
    # 'clip': our method, 'object_detector': Detic baseline
    parser.add_argument('--goal_selector', type=str, default="clip")
    # path of the finetuned CLIP model
    # if you want to use pretrained CLIP model without finetuning, set this argument to None
    parser.add_argument('--clip_model_path', type=str, default='pretrained_models/clip.pt')
    # topic name of realsense D435 camera: /camera/color/image_raw/compressed
    parser.add_argument('--image_topic_name', type=str, default='/camera/color/image_raw/compressed')

    # folder that stores all landmark images and their corresponding poses on map
    parser.add_argument('--landmark_folder', type=str, default='semantic_map/landmark_library')
    # path of pretrained VQA model
    parser.add_argument('--vqa_model_path', type=str, default='pretrained_models/vqa.ckpt')
    args = parser.parse_args()

    parser = NLUIntentParser(args)
    parser.run()
