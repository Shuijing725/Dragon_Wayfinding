# Dragon_Wayfinding

This repository contains the codes for our paper titled "DRAGON: A Dialogue-Based Robot for Assistive Navigation with Visual Language Grounding" in RA-L 2024.   

[[Website]](https://sites.google.com/view/dragon-wayfinding)  [[Paper]](https://ieeexplore.ieee.org/document/10423088) [[arXiv]](https://arxiv.org/abs/2307.06924) [[Videos]](https://www.youtube.com/playlist?list=PLL4IPhbfiY3YkITpyLjeroak_wBn151pn) [[User study documents]](https://drive.google.com/file/d/15KNR6C82mUrKSPMFRCnAJZ1C2NGX7dXJ/view)

------

## Abstract
Persons with visual impairments (PwVI) have difficulties understanding and navigating spaces around them.
Current wayfinding technologies either focus solely on navigation or provide limited communication about the environment.
Motivated by recent advances in visual-language grounding and semantic navigation, we propose DRAGON, a guiding robot powered by a dialogue system and the ability to associate the environment with natural language. 
By understanding the commands from the user, DRAGON is able to guide the user to the desired landmarks on the map, describe the environment, and answer questions from visual observations. 
Through effective utilization of dialogue, the robot can ground the user’s freeform language to the environment, and give the user semantic information through spoken language. 
We conduct a user study with blindfolded participants in an everyday indoor environment.
Our results demonstrate that DRAGON is able to communicate with the user smoothly, provide a good guiding experience, and connect users with their surrounding environment in an intuitive manner.

<p align="center">
<img src="/figures/open_new.png" width="600" />
</p>

------
## Code structure
This repository is organized in five parts: 
- `image_caption/` folder contains the code for object detector, which is used for environment description and baseline landmark recognition.
- `NLU/` folder contains the code for natural language understanding module.
- `semantic_map/` folder contains the code for CLIP.
- `speech_to_text/` contains the code for speech-to-text transcription with OpenAI Whisper. 
- `VQA/` contains code for visual question answering module. 

In most folders, we also provide scripts to unit test each module with and without ROS. See testing files that end with "unit_test_ros.py" and "unit_test.py".

------
## System overview
### Hardware
- Host computer:
  - CPU: Intel i7-9700 @ 3GHz
  - GPU: Nvidia RTX 2080 **(GPU and cuda are necessary for the host computer)**
  - Memory: 32GB
- Turtlebot2i:
  - On-board computer: Nvidia Jetson Xavier
  - Lidar: RP-Lidar A3
  - Camera: Intel Realsense D435
  - Mobile base: Kobuki base
  - Wireless headphone with USB dongle
  - Custom-designed holding point

### Software
The host computer and the turtlebot communicates through ROS by connecting to the same WiFi network.
- Host computer:
  - OS: Ubuntu 20.04
  - Python version: 3.8.10
  - Cuda version: 11.5
  - ROS version: Noetic **(our code WILL NOT WORK with lower versions of ROS or ROS2)**
- Turtlebot2i:
  - OS: Linux
  - Python version: 3.8
  - Cuda version: cuda is not needed unless you're running everything on board (i.e. no host computer)
  - ROS version: Melodic

------

## Setup

### Turtlebot
1. Create a catkin workspace
   ```
   mkdir ~/catkin_ws
   cd catkin_ws
   mkdir -p src
   catkin_make
   cd src
   ```

2. Install ROS packages into your workspace
   ```
   cd ~/catkin_ws/src
   # turtlebot2
   git clone https://github.com/turtlebot/turtlebot.git
   git clone https://github.com/turtlebot/turtlebot_msgs.git
   git clone https://github.com/turtlebot/turtlebot_apps.git
   git clone https://github.com/turtlebot/turtlebot_interactions.git

   # kobuki
   git clone https://github.com/yujinrobot/kobuki_msgs.git

   # RP-Lidar
   git clone https://github.com/Slamtec/rplidar_ros.git

   # audio capture
   git clone https://github.com/ros-drivers/audio_common.git

   # text-to-speech script
   curl -o ros_speak_caption.py https://raw.githubusercontent.com/Shuijing725/Dragon_Wayfinding/main/text-to-speech/ros_speak_caption.py
   
   cd ~/catkin_ws
   catkin_make
   ```

3. Install realsense-ros following [this link](https://jsk-docs.readthedocs.io/projects/jsk_recognition/en/latest/install_realsense_camera.html)
4. Setup Google text-to-speech service with [this link](https://cloud.google.com/text-to-speech/docs/create-audio-text-command-line). Then, install the following
   ```
   pip3 install --user --upgrade google-cloud-texttospeech
   pip install playsound pygobject
   ```

### Host computer
1. Create a catkin workspace
   ```
   mkdir ~/catkin_ws
   cd catkin_ws
   mkdir -p src
   catkin_make
   cd src
   ```

2. Install ROS packages into your workspace
   ```
   cd ~/catkin_ws/src
   # turtlebot2
   git clone https://github.com/turtlebot/turtlebot.git
   git clone https://github.com/turtlebot/turtlebot_msgs.git
   git clone https://github.com/turtlebot/turtlebot_apps.git
   git clone https://github.com/turtlebot/turtlebot_interactions.git

   # kobuki
   git clone https://github.com/yujinrobot/kobuki_msgs.git

   # audio package
   git clone https://github.com/ros-drivers/audio_common.git

   # to use lidar for SLAM
   git clone https://github.com/surfertas/turtlebot2_lidar.git
   git clone https://github.com/SteveMacenski/slam_toolbox.git
   cd slam_toolbox
   git checkout noetic-devel
   rosdep install -q -y -r --from-paths src --ignore-src
   cd ..

   cd ~/catkin_ws
   catkin_make
   ```

3. Create a conda environment for speech recognition with OpenAI Whisper
   ```
   conda env create -f wayfindng.yml
   ```

4. Before proceeding to the next step, **don't active** `wayfinding_new` conda environment you created in Step 4.
   - Everything else below needs to be installed in the SAME environment.
   - To install everything below, we **don't recommend conda environment** since it may have problems with ROS. Instead, we recommend creating a virtual environment or installing everything in root.
5. (For NLU) Install rasa following [this link](https://rasa.com/docs/rasa/installation/installing-rasa-open-source/)
6. (For CLIP) Install [CLIP](https://github.com/openai/CLIP) and its dependencies 
   ```
   pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   pip install ftfy regex tqdm
   git clone https://github.com/openai/CLIP.git
   cd CLIP
   # If you want to finetune CLIP model with your own dataset, you need to install this patch
   git checkout jongwook-patch-1
   pip install -e .
   pip install packaging==21.3
   ```
7. (For baseline landmark recognition and environment description) Install Detic object detector following [this link](https://github.com/facebookresearch/Detic)
8. (For VQA) Install ViLT following [this link](https://github.com/dandelin/ViLT?tab=readme-ov-file)
9. If you want to use our pretrained models, download them from [here](https://drive.google.com/drive/folders/1vmfiVmH2krR42z5066ScQy2LA5gTibLF?usp=sharing), create a folder named `pretrained_models/` in root, and place all models in the folder. 
10. Connect the robot and host computer to the same WiFi network. In `tb2.bash`, change `ROS_MASTER` to the IP address of the robot, change `ROS_IP` to the IP address of the host computer.


------

## Run the code
### Training and Preparation
1. Create a map of the real environment using SLAM. For reference, the map used in our paper are in [`semantic_maps/metric_maps`](https://github.com/Shuijing725/Dragon_Wayfinding/tree/main/semantic_map/metric_maps).   

   a. **[Turtlebot]** Launch the mobile base:
      ```
      source catkin_ws/devel/setup.bash
      roslaunch turtlebot2i_bringup minimal.launch
      ```

   b. **[Turtlebot]** Plug in the USB wire of LiDAR, launch the LiDAR:
      ```
      source catkin_ws/devel/setup.bash && sudo chmod 666 /dev/ttyUSB0 && sudo chmod 666 /dev/ttyUSB1 && sudo chmod 666 /dev/ttyUSB2 
      roslaunch rplidar.launch
      ```
   c. **[Host computer]** Launch SLAM and navigation
      ```
      source tb2.bash
      source ~/catkin_ws/devel/setup.bash
      roslaunch turtlebot_navigation laser_gmapping_demo.launch 
      ```
   d. **[Host computer]** Launch RViz
      ```
      source tb2.bash
      source ~/catkin_ws/devel/setup.bash
      roslaunch turbot_rviz nav.launch
      ```
   e. **[Host computer]** Launch robot teleoperation
      ```
      source tb2.bash 
      roslaunch turtlebot_teleop keyboard_teleop.launch
      ```
   f. **[Host computer]** Teleoperate the robot around the environment until you are satisfied with the map in RViz, save the map by 
      ```
      rosrun map_server map_saver -f ~/map
      ```
      In your home directory, you will see two files: `map.yaml` and `map.pgm`. You can manually edit `map.pgm` to add/remove obstacles if the environment changes.      
      
2. Collect landmark images and poses. For reference, the landmark data in our paper is provided in [`semantic_map/landmark_library`](https://github.com/Shuijing725/Dragon_Wayfinding/tree/main/semantic_map/landmark_library).     
   
   a. **[Turtlebot]** Launch the mobile base (see [Training and preparation](#training-and-preparation) -> Step 1a)

   b. **[Turtlebot]** Launch the LiDAR (see [Training and preparation](#training-and-preparation) -> Step 1b)

   c. **[Host computer]** Launch AMCL localization and navigation
     ```
     source tb2.bash
     source ~/catkin_ws/devel/setup.bash
     roslaunch turtlebot_navigation laser_amcl_demo.launch map_file:=$HOME/map.yaml
     ```
     This step is ready if the terminal shows "odom received".

   d. **[Host computer]** Launch RViz (see [Training and preparation](#training-and-preparation) -> Step 1d)  
      - To correct the initial localization, click "2D pose estimate" to correct the initial pose of robot, and then click "2D navigation" to navigate the robot around until the localization particles converge.  
        The video below demonstrates the calibration process:  
        [![Shuijing Liu on YouTube](http://img.youtube.com/vi/MdZ6RLviqx4/0.jpg)](http://www.youtube.com/watch?v=MdZ6RLviqx4 "Calibrate robot localization in AMCL in ROS navigaion stack")  
   
   e. **[Host computer]** Launch robot teleoperation (see [Training and preparation](#training-and-preparation) -> Step 1e)
      
   f. Run the code to collect (image, robot pose) pairs, 
      ```
      source tb2.bash
      source ~/catkin_ws/devel/setup.bash
      cd ~/Dragon_Wayfinding/semantic_map
      python collect_image_pose.py
      ```
   g. Teleoperate the robot to landmarks of interest, then press "1" on the keyboard to record the (image, robot pose in map frame) when the robot stops at the landmarks. 
      - The data is saved in a folder named with the current date and time. 
      - Press "Ctrl+C" to exit the code when you are done.
   
3. (Optional) If the performance of pretrained CLIP model is not satisfactory, you can collect an image dataset with ground truth landmark labels in step 2, and run [`semantic_map/clip_finetune.py`](https://github.com/Shuijing725/Dragon_Wayfinding/blob/main/semantic_map/clip_finetune.py) to finetune CLIP.
   - Our finetuned CLIP model can be downloaded in [Setup -> Host Computer](#host-computer) -> Step 9.
4. (Optional) Modify the NLU training data and the set of all intents in [`NLU/data/nlu.yml`](https://github.com/Shuijing725/Dragon_Wayfinding/blob/main/NLU/data/nlu.yml), and train NLU:
   ```
   cd ~/dragon_wayfnding/NLU
   rasa train
   ```
   If you want to use our pretrained NLU model, download it in [Setup -> Host Computer](#host-computer) -> Step 9.
5. (Optional) If the performance of pretrained VQA model is not satisfactory, you can collect a dataset of (image, question, answer) triplets, and follow the intructions on [their Github](https://github.com/dandelin/ViLT/blob/master/TRAIN.md#finetune-on-vqav2).

### Testing

a. **[Turtlebot]** Launch the mobile base (see [Run the code](#run-the-code) -> Step 1a)

b. **[Turtlebot]** Launch the LiDAR (see [Training and preparation](#training-and-preparation) -> Step 1b)
 
c. **[Turtlebot]** Launch the camera 
   ```
   cd ~/catkin_ws 
   source devel/setup.bash 
   roslaunch realsense2_camera rs_camera.launch
   ```
d. **[Turtlebot]** Connect the microphone to Turtlebot, and launch audio capture
   ```
   cd ~/catkin_ws 
   source devel/setup.bash 
   roslaunch audio_capture capture.launch
   ```
   To test whether the microphone is working, you can record and play a test audio:
   ```
   # say something to the mic
   arecord -d 5 test-mic.wav 
   # if you hear what you said, the mic is good to go
   aplay test-mic.wav
   ```
e. **[Turtlebot]** Launch text-to-speech
   ```
   cd ~/catkin_ws
   source devel/setup.bash 
   python3 src/ros_speak_caption.py
   ```
f. **[Host computer]** Launch AMCL localization and navigation (see [Training and preparation](#training-and-preparation) -> Step 2c)

g. **[Host computer]** Launch RViz and calibrate localization (see [Training and preparation](#training-and-preparation) -> Step 1d)  

h. **[Host computer]** Launch speech-to-text
   ```
   conda activate wayfinding_new 
   source tb2.bash 
   source ~/catkin_ws/devel/setup.bash
   cd ~/Dragon_Wayfinding/speech-to-text
   python audio_script.py
   ```

i. **[Host computer]** Modify the arguments in line 460-474 in [main.py](https://github.com/Shuijing725/Dragon_Wayfinding/blob/main/main.py). Launch the main function
   ```
   source tb2.bash 
   source ~/catkin_ws/devel/setup.bash
   cd ~/Dragon_Wayfinding 
   python main.py
   ```
   Now you can speak to the robot, and once it parses your command, it will start guiding and other functionalities!

------
## Disclaimer
1. We only tested our code in our system as specified in [System overview](#system-overview). The code may work on other hardware or software, but we do not have any guarantee.  

2. Since there are lots of uncertainties in real-world experiments that may affect performance, we cannot guarantee that our result is reproducible in your case.

------
## Citation
If you find the code or the paper useful for your research, please cite the following papers:
```
@article{liu2024DRAGON,
    title={{DRAGON}: A Dialogue-Based Robot for Assistive Navigation with Visual Language Grounding},
    author={Liu, Shuijing and Hasan, Aamir and Hong, Kaiwen and Wang, Ruxuan and Chang, Peixin and Mizarchi, Zachary and Lin, Justin and McPherson, D. Livingston and Rogers, Wendy A. and Driggs-Campbell, Katherine},
    journal={IEEE Robotics and Automation Letters},
    year={2024},
    volume={9},
    number={4},
    pages={3712-3719}
}
```

------
## Credits
Part of the code is adapted from the following repositories:
- https://github.com/openai/CLIP
- https://github.com/facebookresearch/Detic
- https://github.com/dandelin/ViLT

Other contributors:  
- [Aamir Hasan](https://github.com/aamzhas)  
- [Kaiwen Hong](https://www.linkedin.com/in/kaiwen-hong-524520141/en)   
- [Ruoxuan Wang](https://www.linkedin.com/in/runxuan-wang)  
- [Zachary Mizarachi](https://zachmizrachi.com/)  
- [Peixin Chang](https://github.com/PeixinC)

------

## Contact
If you have any questions or find any bugs, please feel free to open an issue or a pull request.
