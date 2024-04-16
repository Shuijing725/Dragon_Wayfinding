#!/usr/bin/env python
# license removed for brevity

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import clip
import os

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
    def __init__(self, landmark_folder, clip_model_preprocessor=None, custom_clip_model_path=None):
        """
        landmark_folder: path of all (image, pose) pairs
        clip_model_preprocessor: a tuple of preloaded clip model and preprocessor
        """
        self.landmark_folder = landmark_folder
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
        # enable all images to be iterated multiple times
        # self.load_data(image_folder)
        self.img_ds = ImageDataset(image_folder, self.preprocessor)
        self.generator = torch.utils.data.DataLoader(self.img_ds,
                                                     batch_size=len(self.img_ds),
                                                     shuffle=False,
                                                     num_workers=1,
                                                     pin_memory=True,
                                                     drop_last=True
                                                     )
        self.img_num = len(self.img_ds)
        self.best_score = None


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