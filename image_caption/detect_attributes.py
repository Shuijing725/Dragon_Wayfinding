import torch
import pandas as pd
import numpy as np
import os
import copy
import argparse
import clip
from PIL import Image

class AttriDetector():
    def __init__(self, args):
        # convert attributes to prompts: "that is (attribute_value)"
        for key in args.attributes_categorical:
            args.attributes_categorical[key] = args.attributes_categorical[key].replace(',', ',is ')
        self.all_attr_prompt = {}
        # merge two dicts
        args.attributes_binary.update(args.attributes_categorical)
        # store the attribute value with higest score, will be saved in csv
        self.new_df_dict = {}
        for key, val in args.attributes_binary.items():
            attr_list = val.split(',')
            for i in range(len(attr_list)):
                attr_list[i] = ' that ' + attr_list[i]
            self.all_attr_prompt[key] = attr_list
            self.new_df_dict[key] = []

        self.img_dir = args.data_dir
        self.df = pd.read_csv(os.path.join(args.data_dir, 'patch_text_pairs.csv'))
        self.img_file_list = self.df['filename']
        self.obj_name_list = self.df['text_label']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocessor = clip.load("ViT-B/32", device=self.device, jit=False)
        if args.custom_clip_model_path is not None:
            checkpoint = torch.load(args.custom_clip_model_path)
            self.clip_model.load_state_dict(checkpoint['model_state_dict'])

    def add_prefix_to_attr(self, prefix):
        new_prompts = {}
        for key in self.all_attr_prompt:
            cur_attr_prompt = []
            for j in range(len(self.all_attr_prompt[key])):
                cur_attr_prompt.append('A '+prefix+self.all_attr_prompt[key][j])
            new_prompts[key] = copy.deepcopy(cur_attr_prompt)
        return new_prompts

    def detect_attr(self):
        """
        Given a list of image bboxes, determine the attribute value for each binary/categorical attribute using Prompt enginnering + CLIP
        e.x. for binary attributes, the prompts for Clip text branch are "a xxx that is red" and "a xxx that is not red"
        for categorical attributes, the prompts are "a xxx that is red/blue/yellow..."
        """
        for i in range(len(self.img_file_list)):
            image = self.preprocessor(Image.open(os.path.join(self.img_dir, self.img_file_list[i]))).unsqueeze(0).to(self.device)
            all_texts = self.add_prefix_to_attr(self.obj_name_list[i])
            for key, val in all_texts.items():
                # for each attribute, find the value that has the highest CLIP score
                text = clip.tokenize(val).to(self.device)
                with torch.no_grad():
                    logits_per_image, logits_per_text = self.clip_model(image, text)
                    probs_text = logits_per_text.squeeze(-1).cpu().numpy()
                # assign the highest value to the image patch
                best_idx = np.argmax(probs_text)
                self.new_df_dict[key].append(self.all_attr_prompt[key][best_idx].split(' ')[-1])
        new_df = pd.DataFrame(self.new_df_dict)
        new_df = pd.concat([self.df, new_df], axis=1)
        new_df.to_csv(os.path.join(self.img_dir, 'patch_text_attr_pairs.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    # for attribute grounding
    parser.add_argument(
        "--attributes_binary",
        default={}, # {'arm': 'has arms,has no arms'},
        help="",
    )
    parser.add_argument(
        "--attributes_categorical",
        default={'color': 'blue,orange,green,red,purple,brown,black,white,gray,pink,yellow',
                 'shape': 'round,rectangular',
                 'material': 'wooden,leather,farbic'},
        help="",
    )
    parser.add_argument(
        "--data_dir",
        default='/home/shuijing/Desktop/wayfinding/csl_intents/testing_new/detect_custom/bbox_out',
        help="",
    )
    parser.add_argument(
        "--custom_clip_model_path",
        default='/home/shuijing/Desktop/wayfinding/wayfinding/semantic_map/checkpoints/ambiguity_csl_aug_8intents_lr/30.pt',
        help="",
    )

    args = parser.parse_args()
    grounder = AttriDetector(args)
    grounder.detect_attr()