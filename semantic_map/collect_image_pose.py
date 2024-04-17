import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
# from std_msgs.msg import Float32MultiArray
import tf2_ros


import sys
from select import select
import os, datetime
import pickle

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty


class ImageConverter:

    def __init__(self):

        self.bridge = CvBridge()
        self.color_image = None
        self.img_counter = 18
        self.settings = self.saveTerminalSettings()
        self.save_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.save_dir)
        self.img_save_dir = os.path.join(self.save_dir, 'images')
        os.makedirs(self.img_save_dir)
        self.pose_save_dir = os.path.join(self.save_dir, 'poses')
        os.makedirs(self.pose_save_dir)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)


    def saveTerminalSettings(self):
        if sys.platform == 'win32':
            return None
        return termios.tcgetattr(sys.stdin)

    def getKey(self, settings, timeout):
        if sys.platform == 'win32':
            # getwch() returns a string on Windows
            key = msvcrt.getwch()
        else:
            tty.setraw(sys.stdin.fileno())
            # sys.stdin.read() returns a string on Linux
            rlist, _, _ = select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def color_image_callback(self, img_msg):
        # rospy.logdebug('new color_image, timestamp %d', img_msg.header.stamp.secs)
        self.color_image_timestamp = img_msg.header.stamp.secs
        # self.color_image = ros_numpy.numpify(data)
        self.color_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        print('color_image callback')
        key_timeout = rospy.get_param("~key_timeout", 0.5)
        key = self.getKey(self.settings, key_timeout)
        print(key)
        if key == '1':
            # save image
            print('save image No.', self.img_counter)
            filename = str(self.img_counter) + ".png"
            cv2.imwrite(os.path.join(self.img_save_dir, filename), self.color_image)
            # save robot pose
            filename = os.path.join(self.pose_save_dir, str(self.img_counter) + '.pickle')
            trans = self.tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
            x_, y_ = trans.transform.translation.x, trans.transform.translation.y
            x_r, y_r, z_r, w_r = trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w
            # print(x_r, y_r,z_r,w_r)
            with open(filename, 'wb') as f:
                pickle.dump([x_, y_, x_r, y_r, z_r, w_r], f, protocol=pickle.HIGHEST_PROTOCOL)
            self.img_counter = self.img_counter + 1


def start_node():
    rospy.init_node('pose_estimator', log_level=rospy.INFO, disable_signals=True)
    rospy.loginfo('pose_estimator node started')
    
    converter = ImageConverter()

    rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, converter.color_image_callback)
    # hold till system ends
    rospy.spin()


    
if __name__ == '__main__':
    try:
        start_node()

    except rospy.ROSInterruptException:
        exit()



