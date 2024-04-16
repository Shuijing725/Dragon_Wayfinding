#!/usr/bin/env python
# license removed for brevity

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import LaserScan

import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import clip
import os
import tf2_ros


# goal_pose = None

# each data: [image file name, Image object]
class ImageDataset(Dataset):
    def __init__(self, image_folder, preprocessor):
        self.images = []
        self.preprocessor = preprocessor
        for image_file in os.listdir(image_folder):
            if '.png' in image_file:
                processed_img = self.preprocessor(Image.open(os.path.join(image_folder, image_file)))
                self.images.append([image_file, processed_img])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index][0], self.images[index][1]

class SelectGoalClip():
    def __init__(self, landmark_folder, pub, method, clip_model_preprocessor=None, custom_clip_model_path=None):
        """
        landmark_folder: path of all (image, pose) pairs
        clip_model_preprocessor: a tuple of preloaded clip model and preprocessor
        """
        self.landmark_folder = landmark_folder

        # text publisher, used to announce we're already at the goal
        self.text_pub = pub
        self.method = method
        if method == 'clip':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # No clip model is passed in, load one by ourselves
            if clip_model_preprocessor is None:
                # RN50x4, ViT-B/32
                self.clip_model, self.preprocessor = clip.load("ViT-B/32", device=self.device, jit=False)
            else:
                self.clip_model, self.preprocessor = clip_model_preprocessor

            # if load a finetuned checkpoint
            if custom_clip_model_path is not None:
                checkpoint = torch.load(custom_clip_model_path)
                # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
                # checkpoint['model_state_dict']["input_resolution"] = self.clip_model.input_resolution  # default is 224
                # checkpoint['model_state_dict']["context_length"] = self.clip_model.context_length  # default is 77
                # checkpoint['model_state_dict']["vocab_size"] = self.clip_model.vocab_size
                self.clip_model.load_state_dict(checkpoint['model_state_dict'])

            # load images
            image_folder = os.path.join(self.landmark_folder, 'images')
            self.img_ds = ImageDataset(image_folder, self.preprocessor)
            self.generator = torch.utils.data.DataLoader(self.img_ds,
                                                         batch_size=len(self.img_ds),
                                                         shuffle=False,
                                                         num_workers=1,
                                                         pin_memory=True,
                                                         drop_last=True
                                                         )

        elif method == 'object_detector':
            # key: object, value: (detected from i-th image, count in the image)
            self.detections = {'chair': [('3', 1), ('12', 3), ('17', 1), ('32', 2)],
                               'sofa': [('3', 2),],
                               'thermostat': [('3', 1), ],
                               'dispenser': [('8', 1), ('12', 1),],
                               'faucet': [('8', 1)],
                               'bowl': [('8', 1), ('12', 3)],
                               'bottle': [('8', 3), ('12', 4), ('17', 2),('24', 3), ('32', 1)],
                               'cabinet': [('12', 18), ],
                               'dining table': [('12', 1)],
                               'kettle': [('12', 1)],
                               'microwave oven': [('12', 1)],
                               'cup': [('12', 3)],
                               'coffee maker': [('12', 1)],
                               'television set': [('17', 1)],
                               'desk': [('17', 1), ('32', 1)],
                               'computer keyboard': [('17', 1), ('32', 2)],
                               'person': [('18', 1)],
                               'vending machine': [('24', 0),]}
            # sort each value by count of objects
            for key in self.detections:
                self.detections[key] = sorted(self.detections[key], key=lambda x: x[1], reverse=True)
            print(self.detections)
        else:
            raise ValueError("Unknown goal selector!")

        self.current_x, self.current_y, self.current_rz, self.current_rw = None, None, None, None
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        self.best_score = None

        self.wrong_transcriptions = {
            'Last Door': 'glass door',
            'so far': 'sofa',
            'so fair': 'sofa',
            'so fast': 'sofa',
            'so fine': 'sofa',
            'two solvents': 'two sofas',
            'to sovereigns': 'two sofas',
            'cheers': 'chairs',
            'vending washing': 'vending machine',
            'venue machine': 'vending machine',
            'vtubing machine': 'vending machine',
            'a vending washing': 'a vending machine',
            'a venue machine': 'a vending machine',
            'a vtubing machine': 'a vending machine',
            'somewhere to by or drink': 'somewhere to buy a drink',
            'a place to by or drink': 'a place to buy a drink',
            'some place to by or drink': 'some place to buy a drink',
            'award station': 'a workstation',
            'a place to say': 'a place to sit',

        }

        # store a sequence of goal poses to be executed
        # each element: [pos_x, pos_y, theta_x, theta_y, theta_z, theta_w]
        self.waypoint_list = []

        sofa_corner_to_entrance = [1.914, -1.659, 0, 0, 0.811, 0.585]
        sofa_corner_to_hallway = [2.049, -1.576, 0, 0, 0.476, -0.879]
        lab_outer_corner_into_lab = [2.128, -3.944, 0, 0, -0.34, 0.94]
        lab_outer_corner_outof_lab = [2.273, -3.945, 0, 0, 0.951, 0.311]
        lab_inner_corner_into_lab = [3.929, -5.021, 0, 0, 0.566, -0.824]
        lab_inner_corner_outof_lab = [3.9, -5.021, 0, 0, 0.78, 0.626]
        lab_door_outof_lab = [3.09, -4.407, 0, 0, 0.978, 0.207]
        lab_door_into_lab = [3.09, -4.407, 0, 0, -0.207, -0.978]
        cafe_to_lab = [2.422, -2.96, 0, 0, 0.599, 0.801]
        lab_to_cafe = [2.433, -2.989, 0, 0, -0.866, 0.501]


        # todo: hardcoded with iros_map_edit
        self.midpoints = {('entrance', 'cafe'):[[3.044, -0.696, 0, 0, 0.0811, 0.997]],
                          ('cafe', 'entrance'): [[3.042, -0.712, 0, 0, -0.99, 0.118]],
                          ('entrance', 'entrance'): [],
                          ('cafe', 'cafe'): [],
                          ('entrance', 'lab'): [sofa_corner_to_hallway, lab_outer_corner_into_lab, lab_inner_corner_into_lab],
                          ('entrance', 'hallway'): [sofa_corner_to_hallway],
                          ('lab', 'entrance'): [sofa_corner_to_entrance],
                          ('hallway', 'entrance'): [sofa_corner_to_entrance],
                          ('hallway', 'lab'): [lab_outer_corner_into_lab, lab_inner_corner_into_lab],
                          ('cafe', 'lab'): [cafe_to_lab, lab_outer_corner_into_lab, lab_door_into_lab, lab_inner_corner_into_lab],
                          ('lab', 'entrance'):[lab_inner_corner_outof_lab, lab_door_outof_lab, lab_outer_corner_outof_lab, sofa_corner_to_entrance],
                          ('lab', 'hallway'): [lab_inner_corner_outof_lab, lab_door_outof_lab, lab_outer_corner_outof_lab],
                          ('lab', 'cafe'): [lab_inner_corner_outof_lab, lab_door_outof_lab, lab_outer_corner_outof_lab, lab_to_cafe],
                          ('lab', 'lab'): [],
                          ('hallway', 'hallway'): [],
                          ('hallway', 'cafe'): [],
                          ('cafe', 'hallway'): []
                          }

        self.cur_goal_x = None
        self.cur_goal_y = None
        self.cur_goal_wz = None
        self.cur_goal_ww = None

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)


    def current_pose_callback(self, data):
        self.current_x, self.current_y = data.pose.pose.position.x, data.pose.pose.position.y
        self.current_rz, self.current_rw = data.pose.pose.orientation.z, data.pose.pose.orientation.w

    # call this function if we use CLIP as goal selector
    def select_pose_from_clip(self, text):
        text = clip.tokenize([text]).to(self.device)
        for i, (image_filename, image) in enumerate(self.generator):
            image = image.to(self.device)
            self.all_img_names = image_filename
            with torch.no_grad():
                logits_per_image, logits_per_text = self.clip_model(image, text)
                jit_probs_text = logits_per_text.softmax(dim=-1).cpu().numpy()
                # print(image_filename, jit_probs_text)
                best_idx = np.argmax(jit_probs_text[0])
                best_image_no = image_filename[best_idx]
                print('Best matching image:', image_filename[best_idx], ',score prob:', jit_probs_text[0][best_idx])
                self.best_score = jit_probs_text[0][best_idx]
        # remove .png from filename string
        return best_image_no[:-4]

    # call this function if we use object detector (Detic) as goal selector
    def select_pose_from_obj_detector(self, text):
        # print(text in self.detections)
        if text in self.detections:
            # select the image that contains the highest number of objects with name == text
            return self.detections[text][0][0]
        else:
            return -1
    def determine_room_coord(self, x, y):
        '''
        hardcoded with iros_map_edit.pgm
        '''
        if 1.836 * x - 5.267 < y:
            return 'entrance'
        elif 1.75 * x - 8.614 > y:
            if -0.578 * x - 0.573 < y:
                return 'cafe'
            else:
                return 'lab'
        else:
            return 'hallway'


    def determine_room_landmark(self, num):
        if num in ['0', '3']:
            return 'entrance'
        elif num in ['8', '12', '24']:
            return 'cafe'
        elif num in ['17', '32']:
            return 'lab'
        else:
            return 'hallway'


    def list_to_move_base_goal(self, goal_pose_list):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_pose_list[0]
        goal.target_pose.pose.position.y = goal_pose_list[1]
        goal.target_pose.pose.position.z = 0.
        goal.target_pose.pose.orientation.x = goal_pose_list[2]
        goal.target_pose.pose.orientation.y = goal_pose_list[3]
        goal.target_pose.pose.orientation.z = goal_pose_list[4]
        goal.target_pose.pose.orientation.w = goal_pose_list[5]
        return goal

    def send_goal(self, text, wait=True):
        # correct wrong ASR transcriptions
        if text in self.wrong_transcriptions:
            text = self.wrong_transcriptions[text]

        if self.method == 'clip':
            # use CLIP to select the robot goal pose
            landmark_num_str = self.select_pose_from_clip(text)

        else:
            landmark_num_str = self.select_pose_from_obj_detector(text)
            # didn't find any matching object in all images
            if landmark_num_str == -1:
                text_out = 'Sorry, I cannot find the goal. Can you rephrase and say it again, please?'
                print(text_out)
                self.text_pub.publish(text_out)
                return -1

        print(landmark_num_str)
        with open(os.path.join(self.landmark_folder, 'poses', landmark_num_str+'.pickle'), 'rb') as f:
            goal_pose_list = pickle.load(f)
            # goal_pose_list = goal_pose.data
        # print(goal_pose_list)

        self.client.wait_for_server()

        # cancel all previous goals
        # status = self.client.get_goal_status_text()
        # if status != "ERROR: Called get_goal_status_text when no goal is running":
        #     self.client.cancel_goal()
        # print(self.check_status(None))
        # todo: determine the self.waypoint_list, not just goal


        trans = self.tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
        cur_x, cur_y = trans.transform.translation.x, trans.transform.translation.y

        # check if we're already at the goal, if so, tell the user
        cur_rz, cur_rw = trans.transform.rotation.z, trans.transform.rotation.w
        # print(cur_x, cur_y, cur_rw, cur_rz)
        # print(goal_pose_list)
        print(np.linalg.norm([cur_x - goal_pose_list[0], cur_y - goal_pose_list[1]]))
        if np.linalg.norm([cur_x - goal_pose_list[0], cur_y - goal_pose_list[1]]) < 0.2:
            text_out = 'We are already at the goal'
            print(text_out)
            self.text_pub.publish(text_out)
            return -1

        # determine current room (entrance, lab, or cafe)
        cur_room = self.determine_room_coord(cur_x, cur_y)
        # use landmark_num_str to determine the last waypoint
        goal_room = self.determine_room_landmark(landmark_num_str)

        # add intermediate waypoints
        self.waypoint_list.clear()
        for pose in self.midpoints[(cur_room, goal_room)]:
            self.waypoint_list.append(pose)

        # add the last waypoint (the detination)
        self.waypoint_list.append(goal_pose_list)

        # send the first goal
        self.client.cancel_goal()
        new_goal = self.waypoint_list.pop(0)
        move_base_goal = self.list_to_move_base_goal(new_goal)
        self.client.send_goal(move_base_goal)
        print('first goal sent')

        # update current goal
        self.cur_goal_x = new_goal[0]
        self.cur_goal_y = new_goal[1]
        self.cur_goal_wz = new_goal[-2]
        self.cur_goal_ww = new_goal[-1]

        return 0

    def check_status(self, data):
        return self.client.get_state()

    def pause_robot(self):
        self.client.cancel_goal()

    def resume_robot(self):
        move_base_goal = self.list_to_move_base_goal([self.cur_goal_x, self.cur_goal_y, 0, 0, 0, self.cur_goal_wz, self.cur_goal_ww])
        self.client.send_goal(move_base_goal)

    def update_goal(self, data):
        self.client.wait_for_server()
        # print(len(self.waypoint_list))
        # if there are remaining waypoints
        if len(self.waypoint_list) > 0:
            trans = self.tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
            cur_x, cur_y = trans.transform.translation.x, trans.transform.translation.y
            # if current pose and next goal is within 0.5m

            if np.linalg.norm([self.cur_goal_x - cur_x, self.cur_goal_y - cur_y]) < 0.3:
                self.client.cancel_goal()
                new_goal = self.waypoint_list.pop(0)
                move_base_goal = self.list_to_move_base_goal(new_goal)

                # send new goal
                self.client.send_goal(move_base_goal)
                print('new goal sent')

                # update current goal
                self.cur_goal_x = new_goal[0]
                self.cur_goal_y = new_goal[1]
                self.cur_goal_wz = new_goal[-2]
                self.cur_goal_ww = new_goal[-1]

if __name__ == '__main__':
    rospy.init_node('movebase_client_py')

    goal_sender = SelectGoalClip(landmark_folder='/home/shuijing/Desktop/var_wayfinding/2023-01-29_entrance_cafe_lab_select', # '/home/shuijing/Desktop/var_wayfinding/pavillon_landmarks',
                                 custom_clip_model_path='checkpoints/checkpoints_csl_aug_8intents_lr/30.pt')
    rospy.Subscriber("/scan", LaserScan, goal_sender.update_goal)

    # language input is hardcoded here
    text = 'an exit door'
    goal_sender.send_goal(text, wait=True)
    rospy.spin()
