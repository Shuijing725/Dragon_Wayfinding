"""
Put this file in the local desktop with GPU, and launch realsense camera before running it
Dependencies: clipclap_modules.py
Subscribe: camera RGB images
Publish: text caption of the current image
"""
import clip
import PIL
import os
import torch
import cv2
import numpy as np

import image_caption.demo as demo

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String


class ImageCaptioner:

    def __init__(self, pub=None):
        self.publisher = pub
        self.bridge = CvBridge()
        self.color_image = None
        self.img_counter = 0
        self.save_dir = os.path.join(os.getcwd(), 'image_temp')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = demo.get_parser().parse_args()
        self.args.config_file = os.path.join(os.getcwd(), self.args.config_file)
        self.args.opts[1] = os.path.join(os.getcwd(), self.args.opts[1])
        cfg = demo.setup_cfg(self.args)

        self.predictor = demo.VisualizationDemo(cfg, self.args)

        self.counter = 0


    def namelist_to_sentence(self, name_list):
        '''
        Convert a list of object names to a sentence description
        '''
        sentence = ""
        counter = 0
        for obj in np.unique(name_list):
            num_occur = name_list.count(obj)
            # and 4 books.
            if counter == len(np.unique(name_list)) - 1 and counter > 0:
                if num_occur > 1:
                    suffix = 's.'
                else:
                    suffix = '.'
                sentence = sentence + 'and ' + str(num_occur) + ' ' + obj + suffix
            else: # 4 books,
                if num_occur > 1:
                    suffix = 's, '
                else:
                    suffix = ', '
                sentence = sentence + str(num_occur) + ' ' + obj + suffix
            counter = counter + 1
        return sentence


    def image_captioning_callback(self, img_msg):
        # convert ros image to PIL image
        self.color_image_timestamp = img_msg.header.stamp.secs
        self.color_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # save the image for debugging
        filename = str(self.img_counter) + ".png"
        cv2.imwrite(os.path.join(self.save_dir, filename), self.color_image)

        img = demo.read_image(os.path.join(self.save_dir, filename), format="BGR")

        predictions, visualized_output = self.predictor.run_on_image(img, self.args)

        # save camera images with bboxes for debugging
        print(os.path.join(self.save_dir, str(self.counter)+'.png'))
        visualized_output.save(os.path.join(self.save_dir, str(self.counter)+'.png'))
        self.counter = self.counter + 1

        # convert predictions to texts
        text = self.namelist_to_sentence(self.predictor.visualizer.labels)
        print(text)

        if self.publisher is not None:
            self.publisher.publish(text)




if __name__ == '__main__':
    try:
        rospy.init_node('image_captioning', log_level=rospy.INFO, disable_signals=True)
        rospy.loginfo('image_captioning node started')

        # publisher for the caption text
        caption_pub = rospy.Publisher('/image_caption_text', String, queue_size=2)

        converter = ImageCaptioner(pub=caption_pub)

        while not rospy.is_shutdown():
            data = rospy.wait_for_message('/camera/color/image_raw/compressed', CompressedImage, timeout=5)
            converter.image_captioning_callback(data)
            rospy.sleep(5)

    except rospy.ROSInterruptException:
        exit()
