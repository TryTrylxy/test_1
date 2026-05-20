import json
import os
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from clip import clip
from .utils import pre_caption
import pandas as pd
import random


class cross_coco_dataset(Dataset):
    def __init__(self, root, transform=None, split="train", max_words=64, config=None):
        self.root = root
        self.transform = transform
        self.split = split
        self.max_words = max_words
        self.config = config

        if 'CUB' not in root:
            self.dataPath = os.path.join(self.root, "new_{}.json".format(self.split))
            with open(self.dataPath, "r", encoding="utf8") as f:
                self.dataList = json.load(f)
        else:
            print(torch.load(root + '/metadata.pth').keys())
            data_imgs = torch.load(root + '/imgs_train_val_256x256.pth')
            data_meta = torch.load(root + '/metadata.pth')
            data_ids_train_val = torch.load(root + '/metadata.pth')['train_val_img_ids']
            self.dataList = []
            annotations = pd.read_csv(root + '/res.json', sep='\t', header=None)
            caps = np.array(annotations)
            for i in range(len(data_ids_train_val)):
                for k in range(len(data_meta['img_id_to_encoded_caps'][data_ids_train_val[i]])):
                    data_cap = [data_meta['word_id_to_word'][j] for j in data_meta['img_id_to_encoded_caps'][data_ids_train_val[i]][k] if j != 717]
                    tmp = {
                        'image_id': data_ids_train_val[i],
                        'caption': ' '.join(data_cap),
                        'caption_r': caps[i][0],
                        'class_id': data_meta['img_id_to_class_id'][data_ids_train_val[i]],
                        'img': data_imgs[i]
                    }
                    self.dataList.append(tmp)
            arg = [(i + t) * 10 + u for i in range(0, len(self.dataList) // 10 , 10) for t in range(9) for u in range(10)]
            self.dataList = [self.dataList[t] for t in arg if t < len(self.dataList)]
        self.dataList = self.dataList[:int(len(self.dataList) * 0.1)]
            
        self.img_ids = {}
        n = 0
        for ann in self.dataList:
            if 'CUB' not in root:
                img_id = ann["image_id"]
            else:
                img_id = ann["class_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        if self.split == "experiment":
            self.split = "train"
        try:
            self.unicom_fea = np.load(os.path.join(self.root, "{}_unicom.npy".format(self.split)), allow_pickle=True).item()
        except:
            self.unicom_fea = None

    def __len__(self):
        # return 200
        return len(self.dataList)

    def __getitem__(self, index):
        tmpData = self.dataList[index].copy()
        if 'VL-check' in self.root:
            if type(tmpData["caption"]) is list:
                tmpData["caption"] = self.dataList[index]["caption"][random.randint(0, len(self.dataList[index]) - 1)]
                tmpData["caption_r"] = self.dataList[index]["caption"][random.randint(0, len(self.dataList[index]) - 1)]
            else:
                tmpData["caption_r"] = tmpData["caption"]
        
        caption = pre_caption(tmpData["caption"], self.max_words)

        raw_caption = caption
        caption = clip.tokenize(caption)[0]
        if self.config['do_LLMs_ab']:
            raw_caption_r = pre_caption(tmpData["caption"], self.max_words)
        else:
            raw_caption_r = pre_caption(tmpData["caption_r"], self.max_words)
        if self.config['do_neg']:
            caption_neg = pre_caption(tmpData["caption_neg"], self.max_words)
        else:
            caption_neg = ""
        caption_neg = clip.tokenize(caption_neg)[0]

        image_feature = torch.tensor([0.0])
        if self.unicom_fea is not None:
            image_feature = self.unicom_fea.get(tmpData["image_id"])

        if 'CUB' not in self.root:
            im = Image.open(os.path.join(self.root, tmpData["image_path"])).convert('RGB')
            im_neg = Image.open(tmpData["image_path_neg"]).convert('RGB')
        else:
            np_array = tmpData['img'].permute(1, 2, 0).numpy()
            im = Image.fromarray(np_array)
        im = self.transform(im)
        im_neg = self.transform(im_neg)
        if 'CUB' not in self.root:
            return im, caption, image_feature, raw_caption, self.img_ids[tmpData["image_id"]], raw_caption_r, caption_neg, im_neg
        else:
            return im, caption, image_feature, raw_caption, self.img_ids[tmpData["class_id"]], raw_caption, caption_neg
            # return im, caption, image_feature, raw_caption, self.img_ids[tmpData["class_id"]], raw_caption_r


class cross_coco_test_dataset(Dataset):
    def __init__(self, root, transform=None, split="test", max_words=64, i=None, num_data=None):
        self.transform = transform
        self.dataPath = root + "/new_{}.json".format(split)

        with open(self.dataPath, "r", encoding="utf8") as f:
            self.dataList = json.load(f)[:num_data]
        self.text = [x['caption'] for x in self.dataList]
        self.text_neg = [x['caption_neg'] for x in self.dataList]
        self.img_path = []
        self.img_path_neg = []
        for i in range(len(self.dataList)):
            if self.dataList[i]['image_path'] not in self.img_path:
                self.img_path.append(self.dataList[i]['image_path'])
                self.img_path_neg.append(self.dataList[i]['image_path_neg'])
        self.i2t_label = [[] for _ in range(len(self.img_path))]
        self.t2i_label = []
        for i in range(len(self.dataList)):
            self.i2t_label[self.img_path.index(self.dataList[i]['image_path'])].append(i)
            self.t2i_label.append([self.img_path.index(self.dataList[i]['image_path'])])

    def preprocess_text(self, textList):
        preCaptionList = clip.tokenize(textList, truncate=True)
        return preCaptionList

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        im = Image.open(self.img_path[index]).convert('RGB')
        im = self.transform(im)
        im_neg = Image.open(self.img_path_neg[index]).convert('RGB')
        im_neg = self.transform(im_neg)
        return im, im_neg, index