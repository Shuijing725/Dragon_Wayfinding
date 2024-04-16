# wayfinding

# after installing zed:
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
baselines 0.1.6 requires gym<0.16.0,>=0.15.4, which is not installed.
tensorflow 2.8.4 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.0 which is incompatible.
rasa 3.4.2 requires numpy<1.24.0,>=1.19.2; python_version >= "3.8" and python_version < "3.11", but you have numpy 1.24.4 which is incompatible.
rasa 3.4.2 requires packaging<21.0,>=20.0, but you have packaging 21.3 which is incompatible.
rasa 3.4.2 requires prompt-toolkit<3.0.29,>=3.0, but you have prompt-toolkit 3.0.38 which is incompatible.
rasa 3.4.2 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.0 which is incompatible.
imageio 2.25.0 requires pillow>=8.3.2, but you have pillow 8.2.0 which is incompatible.

## Recording location
- Human speech: /home/shuijing/Desktop/wayfinding/speech-to-text/ros_src/speech_recog/src
- Robot speech: /home/shuijing/data
- Robot camera: /home/shuijing/Desktop/wayfinding/image_temp

## How to run the whole pipeline
- Preparation:
  - Install the base battery and the computer battery
  - Unplug the lidar wire
  - Turn on the computer battery and the base
  - Install D435 camera, the handle (please fix the handle as tightly as possible), and the microphone
- Log into local computer (username: csl, password: 123456)
- Launch commands on Turtlebot:
  - open a terminal, type "ssht", you will see a new terminal remotes into the turtlebot computer
  - in the remote terminal, type "init" to launch base
  - replug the lidar wire
  - in the remote terminal, type "hardware" to launch lidar, camera, microphone, and audio capture
- Launch commands on local computer:
  - open a terminal, type "rosrun map_server map_server iros_map_edit.yaml"
  - open a terminal, type "laser_amcl", in the first step, type "/home/shuijing/iros_map_edit.yaml"
    - Only hit "Enter" twice to launch amcl and rviz, DON'T LAUNCH TELEOP!!!
  - In "/home/shuijing/Desktop/wayfinding/main.py", change the parameters in the bottom of the file
    - set "goal_selector"
  - type "speech" to launch speech recognition and main function, both scripts can take a while to start running

## Todo list

Wayfinding for People with Visual Impairments

Developing a robot helper to assist PwVI in navigating indoor spaces.

- Install Detic:
  - Install Detectron2: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
    - `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
  - Install Detic following https://github.com/facebookresearch/Detic/blob/main/docs/INSTALL.md
- To make CLIP compatible with Pytorch 1.8.0:
  - IMPORTANT!!!!!! DO THIS FIRST!!!!!!
    - save `pip list` of current environment to a file for backup
  - Install pytorch with 
    `pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
  - Re-install CLIP with the patch from branch `jongwook-patch-1`
    - cd into the CLIP repo
    - `git checkout jongwook-patch-1`
    - `pip install -e .`
  - If "ImportError: cannot import name 'LegacyVersion' from 'packaging.version'" error occurs, run
    - `pip install packaging==21.3`
