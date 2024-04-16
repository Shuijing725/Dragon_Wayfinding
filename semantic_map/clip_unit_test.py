import pandas as pd
import os
import sys
import logging
import numpy as np
from semantic_map.set_goal_pose_clip_no_ros import *
from semantic_map.clip_finetune import description_to_command

# test_csv_file = '/home/shuijing/Desktop/wayfinding/pavillon_landmarks/images/pavillon.csv'
test_csv_file = '/home/shuijing/Desktop/wayfinding/csl_intents/testing_new/ambiguity_description.csv'
# test_csv_file = '/home/shuijing/Desktop/wayfinding/csl_intents/testing_new/landmarks_descriptions.csv'
goal_sender = SelectGoalClip(landmark_folder='/home/shuijing/Desktop/wayfinding/csl_intents/testing_new',
                             custom_clip_model_path='/home/shuijing/Desktop/wayfinding/wayfinding/semantic_map/checkpoints/ambiguity_csl_aug_8intents_lr/30.pt'
                             )

# initialize the log
log_path = os.path.join(os.getcwd(), 'test')
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_file = os.path.join(log_path, 'csl_finetune_aug_8intents_lr_ambiguity_30.pt.log')
file_handler = logging.FileHandler(log_file, mode='w')
stdout_handler = logging.StreamHandler(sys.stdout)
level = logging.INFO
logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                    format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


correct_count_all = 0.
correct_count_per_img = np.zeros(goal_sender.img_num)
correct_score_all = 0.
correct_score_pre_img = np.zeros(goal_sender.img_num)


# load descriptions from csv
df = pd.read_csv(test_csv_file)
for i in range(len(df['Description'])):
    # sentence = description_to_command(df['Description'][i])
    sentence = df['Description'][i]
    label = df['Filename'][i]
    best_img = goal_sender.select_pose_from_clip(sentence)
    idx = goal_sender.all_img_names.index(best_img+'.png')
    print(sentence)
    # selection is correct
    if int(best_img) == label:
        correct_count_all = correct_count_all + 1
        correct_count_per_img[idx] = correct_count_per_img[idx] + 1
        correct_score_all = correct_score_all + goal_sender.best_score
        correct_score_pre_img[idx] = correct_score_pre_img[idx] + goal_sender.best_score
        print('correct\n')
    else:
        print('wrong, true image:',label, '\n')
    # just for testing
    # if i > 5:
    #     break

print('correct_count_all', correct_count_all)
print('correct_count_per_img', correct_count_per_img)
print('correct_score_all', correct_score_all)
print('correct_score_pre_img', correct_score_pre_img)

accuracy_all = correct_count_all / len(df['Description'])
# we have 5 descriptions for each image
print('number of pairs:', len(df['Description']))
num_text_per_image = len(df['Description']) / goal_sender.img_num
accuracy_per_img =  correct_count_per_img / num_text_per_image
# average score for all correct (image, text) pairs
correct_score_all = correct_score_all / correct_count_all
# correct_score_pre_img = correct_score_pre_img / correct_count_per_img

# log the results
logging.info('Average accuracy of all images: {:.2f}, average scores of all correct images: {:.2f}'.format(accuracy_all, correct_score_all))
for i in range(goal_sender.img_num):
    if correct_count_per_img[i] == 0:
        avg_score = 0
    else:
        avg_score = correct_score_pre_img[i] / correct_count_per_img[i]
    logging.info('Image {}: average accuracy: {:.2f}, average scores for correct descriptions: {:.2f}'
                 .format(goal_sender.all_img_names[i], accuracy_per_img[i], avg_score))