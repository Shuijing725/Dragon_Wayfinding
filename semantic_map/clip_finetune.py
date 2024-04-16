import copy

import numpy as np
import torch,os

import clip
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import nn
from torch import optim

from PIL import Image
from PIL import ImageFilter, ImageEnhance

import pandas as pd


def description_to_command(description):
    # remove period
    # print(description)
    if description[-1] == '.':
        description = description[:-1]
    rand_num = np.random.uniform(0, 8)

    command_prefix = ['Go to', 'Take me to', 'Escort me to', 'Direct me to', 'Guide me to', 'Find a path to', 'Find a way to', 'Lead me to',
                      '', 'To', 'Just take me to', 'Move me to', "I'd really like to go to", "Let's go to", "Show me the way to"]
    question_prefix = ['Can you take me to', 'Where is', 'Can show me to', 'How could I go to', 'How could we go from here to',
                 'How do I get to', 'Can I be taken to', 'Can I be directed to', 'Is there a route to',
                 'Would you mind taking me to', 'Would you mind kindly taking me to', 'What about going to', 'Could we pop by']
    gratitute_postfix = ['', '',  '', '', '', '', '', '', ' Thank you!', ' Thanks.']
    j = np.random.choice(gratitute_postfix)
    if rand_num <= 1:
        sentence = description + j
    elif 1 < rand_num <= 2:
        first = command_prefix
        i = np.random.choice(first)
        sentence = i + ' ' + description + j
    elif 2 < rand_num <= 3:
        first = command_prefix
        i = np.random.choice(first)
        sentence = i + ' ' + description + ', please.' + j
    elif 3 < rand_num <= 4:
        first = command_prefix
        i = np.random.choice(first)
        sentence = i + ' ' + description + ' now.' + j
    elif 4 < rand_num <= 5:
        first = command_prefix
        i = np.random.choice(first)
        sentence = 'Please ' + i + ' ' + description + '.' + j
    elif 5 < rand_num <= 6:
        first = copy.deepcopy(question_prefix)
        first.extend(['Could you please take me to', 'Would you please go to'])
        i = np.random.choice(first)
        sentence = i + ' ' + description + '?' + j
    elif 6 < rand_num <= 7:
        first = copy.deepcopy(question_prefix)
        first.extend(['Could you please take me to', 'Would you please go to'])
        i = np.random.choice(first)
        j = np.random.choice(gratitute_postfix)
        sentence = i + ' ' + description + '?' + j
    else:
        first = question_prefix
        i = np.random.choice(first)
        sentence = i + ' ' + description + ', please?' + j
    return sentence

class ImageTextDataset(Dataset):
    def __init__(self, image_folders, text_files, preprocessor, to_command=False):
        self.images = []
        self.texts = []
        self.preprocessor = preprocessor
        assert len(image_folders) == len(text_files)

        for j in range(len(image_folders)):
            df = pd.read_csv(text_files[j])

            for i in range(len(df['Description'])):
                # randomly load half of the dataset
                # if the description is not empty
                if not isinstance(df['Description'][i], str):
                    # print('empty:', image_folders[j], df['Filename'][i])
                    continue
                else:
                    # find and preprocess the image
                    image_file = str(df['Filename'][i]) + '.png'
                    processed_img = self.augmentation(Image.open(os.path.join(image_folders[j], image_file)))

                    self.images.append(processed_img)
                    # find and tokenize the description
                    if to_command:
                        text = description_to_command(df['Description'][i])
                    else:
                        text = df['Description'][i]
                    tokenized_text = clip.tokenize(text)
                    self.texts.append(tokenized_text.squeeze(0))

    def augmentation(self, img):
        rand_num = np.random.choice([1, 2, 3, 4])
        # blur
        if rand_num == 1:
            return img.filter(ImageFilter.BLUR)
        # horizontal flip
        elif rand_num == 2:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        elif rand_num == 3:
            enhancer = ImageEnhance.Contrast(img)
            enhancer.enhance(np.random.uniform(0, 2))
            return enhancer.image
        else:
            return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.preprocessor(self.images[index]), self.texts[index]


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)

if __name__ == '__main__':
    BATCH_SIZE = 25
    EPOCH = 50

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training, "ViT-B/32"
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('jit trainable params no:', params)
    # re-initialize parameters
    # reset_model_weights(model)

    # freeze the preprocessor
    # for param in preprocess.transforms[4].parameters():
    #     param.requires_grad = False
    save_dir = os.path.join(os.getcwd(), 'dummy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # use your own data
    image_folder = [
                    '/home/shuijing/Desktop/wayfinding/csl_intents/0',
                    '/home/shuijing/Desktop/wayfinding/csl_intents/3',
                    '/home/shuijing/Desktop/wayfinding/csl_intents/8',
        '/home/shuijing/Desktop/wayfinding/csl_intents/12',
                    '/home/shuijing/Desktop/wayfinding/csl_intents/17',
        '/home/shuijing/Desktop/wayfinding/csl_intents/18',
        '/home/shuijing/Desktop/wayfinding/csl_intents/24',
        '/home/shuijing/Desktop/wayfinding/csl_intents/32',
        '/home/shuijing/Desktop/wayfinding/csl_intents/sink_table',
        '/home/shuijing/Desktop/wayfinding/csl_intents/TV_desk',
        '/home/shuijing/Desktop/wayfinding/csl_intents/vending_hallway_sofa',
                    ]
    text_folder = [
                   '/home/shuijing/Desktop/wayfinding/csl_intents/0/0.csv',
                    '/home/shuijing/Desktop/wayfinding/csl_intents/3/3.csv',
        '/home/shuijing/Desktop/wayfinding/csl_intents/8/8.csv',
                    '/home/shuijing/Desktop/wayfinding/csl_intents/12/12.csv',
                    '/home/shuijing/Desktop/wayfinding/csl_intents/17/17.csv',
        '/home/shuijing/Desktop/wayfinding/csl_intents/18/18.csv',
        '/home/shuijing/Desktop/wayfinding/csl_intents/24/24.csv',
        '/home/shuijing/Desktop/wayfinding/csl_intents/32/32.csv',
        '/home/shuijing/Desktop/wayfinding/csl_intents/sink_table/sink_table.csv',
        '/home/shuijing/Desktop/wayfinding/csl_intents/TV_desk/TV_desk.csv',
        '/home/shuijing/Desktop/wayfinding/csl_intents/vending_hallway_sofa/vending_hallway_sofa.csv',
                   ]
    dataset = ImageTextDataset(image_folder, text_folder, preprocess, to_command=False)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # Define your own dataloader
    print('size of dataset:', len(dataset))
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=8e-6, betas=(0.9, 0.98), eps=1e-6,
                           weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # add your own code to track the training progress.
    for epoch in range(EPOCH):
        for batch in train_dataloader:
            print(model.logit_scale)
            optimizer.zero_grad()

            images, texts = batch

            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            print('total loss:', total_loss.data)
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, os.path.join(save_dir, str(epoch)+".pt"))  # just change to your preferred folder/filename