import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch import nn
import torch.nn.functional as F

input_file = "./data/"
df = pd.read_csv('./data/all_the_image.csv')
df.columns = ['id', 'image', 'caption', 'caption_number']
df['caption'] = df['caption'].astype('string')
df['image'] = df['image'].astype('string')
df['caption'] = df['caption'].str.lstrip()
df['caption_number'] = df['caption_number'].astype('string')
df['caption_number'] = df['caption_number'].str.lstrip()
# ids = [id_ for id_ in range(len(df) // 5) for _ in range(5)]
# df['id'] = ids
df.to_csv("./data/captions.csv", index=False)
image_path = "./data/picture"
captions_path = "./data"
print(df.head(1))

class CFG:
    debug = False
    image_path = image_path
    captions_path = captions_path
    batch_size = 8
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()