import torch
import json
import os

from PIL import Image

from VQA.vilt.modules import ViLTransformerSS
from VQA.vilt.transforms import pixelbert_transform
from VQA.vilt.datamodules.datamodule_base import get_pretrained_tokenizer


class VQAInference():
    def __init__(self, _config):
        self.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

        f = open(os.path.join(os.getcwd(), 'VQA/vqa_dict.json'), 'r')
        self.id2ans = json.loads(f.read())
        f.close()

        self.model = ViLTransformerSS(_config)
        self.model.setup("test")
        self.model.eval()

        self.device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
        self.model.to(self.device)

    def infer(self, image, text):
        # print(text)
        img = pixelbert_transform(size=384)(image)
        img = img.unsqueeze(0).to(self.device)

        batch = {"text": [text], "image": [img]}

        with torch.no_grad():
            encoded = self.tokenizer(batch["text"])
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(self.device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(self.device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(self.device)
            infer = self.model.infer(batch)
            vqa_logits = self.model.vqa_classifier(infer["cls_feats"])

        answer = self.id2ans[str(vqa_logits.argmax().item())]

        return answer


if __name__ == '__main__':
    config = {'exp_name': 'vilt', 'seed': 0, 'datasets': ['coco', 'vg', 'sbu', 'gcc'],
              'loss_names': {"itm": 0, "mlm": 0, "mpp": 0, "vqa": 1, "imgcls": 0, "nlvr2": 0, "irtr": 0, "arc": 0},
              'batch_size': 4096,
              'train_transform_keys': ['pixelbert'], 'val_transform_keys': ['pixelbert'], 'image_size': 384, 'max_image_len': -1,
              'patch_size': 32, 'draw_false_image': 1, 'image_only': False, 'vqav2_label_size': 3129, 'max_text_len': 40,
              'tokenizer': 'bert-base-uncased', 'vocab_size': 30522, 'whole_word_masking': False, 'mlm_prob': 0.15, 'draw_false_text': 0,
              'vit': 'vit_base_patch32_384', 'hidden_size': 768, 'num_heads': 12, 'num_layers': 12, 'mlp_ratio': 4, 'drop_rate': 0.1,
              'optim_type': 'adamw', 'learning_rate': 1e-06, 'weight_decay': 0.01, 'decay_power': 1, 'max_epoch': 100, 'max_steps': 25000,
              'warmup_steps': 2500, 'end_lr': 0, 'lr_mult': 1, 'get_recall_metric': False, 'resume_from': None, 'fast_dev_run': False,
              'val_check_interval': 1.0, 'test_only': True, 'data_root': '', 'log_dir': 'result', 'per_gpu_batchsize': 0, 'num_gpus': 1,
              'num_nodes': 1, 'load_path': '/home/shuijing/Desktop/wayfinding/VQA_training/ViLT/weights/vilt_vqa.ckpt',
              'num_workers': 8, 'precision': 16}

    vqa_module = VQAInference(config)
    image = '/home/shuijing/Desktop/wayfinding/csl_data/image_landmarks/images/7.png'
    image = Image.open(image).convert("RGB")
    text = 'are there any trash cans'
    print(vqa_module.infer(image, text))