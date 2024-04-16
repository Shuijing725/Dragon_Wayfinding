#!/bin/bash
source ~/venv/ros_venv/bin/activate

#for i in 0 3 8 12 17 18 24 32 sink_table TV_desk vending_hallway_sofa;
#do
#python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input /home/shuijing/Desktop/wayfinding/csl_intents/$i/* --output /home/shuijing/Desktop/wayfinding/csl_intents_detect/$i --bbox_output_dir /home/shuijing/Desktop/wayfinding/csl_intents_detect/$i/bbox --vocabulary custom --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
#done

for i in cafe entrance lab;
do
# python demo.py --confidence-threshold 0.7 --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input /home/shuijing/Desktop/wayfinding/csl_data/image_for_captioning/$i/* --output /home/shuijing/Desktop/wayfinding/csl_data/image_for_captioning_detect/$i --bbox_output_dir /home/shuijing/Desktop/wayfinding/csl_data/image_for_captioning_detect/$i/bbox --vocabulary custom --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
python demo.py --confidence-threshold 0.5 --config-file configs/Detic_LCOCOI21k_CLIP_R18_640b32_4x_ft4x_max-size.yaml --input /home/shuijing/Desktop/wayfinding/csl_data/image_for_captioning/$i/* --output /home/shuijing/Desktop/wayfinding/csl_data/image_for_captioning_detect_R18/$i --bbox_output_dir /home/shuijing/Desktop/wayfinding/csl_data/image_for_captioning_detect_R18/$i/bbox --vocabulary custom --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R18_640b32_4x_ft4x_max-size.pth

done