import rospy
import dynamic_reconfigure.client

import numpy as np

class SpeedChanger():
    def __init__(self):
        # for changing the max speed
        self.vel_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS", timeout=2)
        # print(self.vel_client.get_configuration(timeout=2))
        all_params = self.vel_client.get_configuration(timeout=2)
        print('got client')
        self.cur_trans_vel = all_params['max_vel_trans']
        self.max_trans_vel = 0.5
        self.min_trans_vel = 0.15

        self.cur_rot_vel = all_params['max_vel_theta']
        self.max_rot_vel = 5.0
        self.min_rot_vel = 1.0
        print('intial v:', self.cur_trans_vel, 'initial w:', self.cur_rot_vel)

    def change_speed(self, faster):
        """
        faster = True: speed up
        faster = False: slow down
        """
        # https://answers.ros.org/question/359120/modifying-parameter-values-rqt_reconfigure-using-a-python-script/
        # check which params to change
        if faster:
            print('speed up')
            # new_trans_vel = self.cur_trans_vel + 0.1
            new_rot_vel = self.cur_rot_vel + 1.
        else:
            print('speed down')
            # new_trans_vel = self.cur_trans_vel - 0.1
            new_rot_vel = self.cur_rot_vel - 1.
        # self.cur_trans_vel = np.clip(new_trans_vel, self.min_trans_vel, self.max_trans_vel)
        self.cur_rot_vel = np.clip(new_rot_vel, self.min_rot_vel, self.max_rot_vel)

        print('new v:', self.cur_trans_vel, 'new w:', self.cur_rot_vel)
        self.vel_client.update_configuration({'max_vel_trans': self.cur_trans_vel, 'max_vel_theta': self.cur_rot_vel})


rospy.init_node("main")
speed_changer = SpeedChanger()
rospy.sleep(2)
speed_changer.change_speed(False)
rospy.sleep(3)
speed_changer.change_speed(False)