"""
在 LAMA 三个数据集（Attribute, Object, Relation）上测试模型性能的脚本。

使用说明：
---------
python test_lama_datasets.py \
  --checkpoint "/path/to/your/checkpoint.ckpt" \
  --config "./configs/vitb32/flickr/cusa.yaml" \
  --gpu "cuda:0"

配置选择依据：
-----------
1. 默认使用 cusa.yaml 作为基础配置，因为：
   - 你的 checkpoint 文件名：CLIP_ViT-B32_finetune_ALL_LoraRank16_seed0.ckpt
   - "CLIP_ViT-B32" 表明 backbone="CLIP", clip_model="ViT-B/32"
   - cusa.yaml 是最基础的配置，没有各种 ablation 设置
   
2. 如果你的模型是用特殊配置训练的（如 cusa_restate_sl.yaml），请通过 --config 指定

3. Checkpoint 中没有保存 config 信息，所以需要手动提供配置文件
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from clip import clip
from tqdm import tqdm
import yaml

from dataset.cross_coco_dataset import cross_coco_test_dataset
from evaluation import evaluation, itm_eval
from unire.model import unire


class LamaTestDataset(Dataset):
    """自定义数据集类，用于加载 lama 的三个数据集"""
    def __init__(self, root, transform=None, split="test"):
        self.root = root
        self.transform = transform
        self.split = split
        
        self.dataPath = os.path.join(self.root, "new_test.json")
        with open(self.dataPath, "r", encoding="utf8") as f:
            self.dataList = json.load(f)
        
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


def create_loader(dataset, batch_size=128, num_workers=8):
    """创建数据加载器"""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    return loader


def test_on_dataset(model, dataset_name, dataset_path, device, args, config):
    """在单个数据集上进行测试"""
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name} dataset...")
    print(f"{'='*60}")
    
    dataset = LamaTestDataset(dataset_path, transform=model.preprocess)
    data_loader = create_loader(dataset, batch_size=config['batch_size_testall'], num_workers=args.num_workers)
    
    score_i2t, score_t2i, score_i2i, score_t2t, count_text, count_image = evaluation(
        model, data_loader, device, args
    )
    
    result = itm_eval(
        config, score_i2t, score_t2i, score_i2i, score_t2t,
        data_loader, device=device, flag=False
    )
    result['text_neg'] = count_text
    result['image_neg'] = count_image
    
    return result


def main(args, config):
    device = torch.device(args.gpu)
    
    print("Loading checkpoint from %s" % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    
    print("Creating model")
    model = unire(args, config)
    msg = model.load_state_dict(state_dict)
    print(msg)
    model.to(device)
    model.eval()
    
    datasets_config = [
        ("Attribute", "/home/jncsnlp3/SSD2/syy/lama-main/dataset/Attribute"),
        ("Object", "/home/jncsnlp3/SSD2/syy/lama-main/dataset/Object"),
        ("Relation", "/home/jncsnlp3/SSD2/syy/lama-main/dataset/Relation"),
    ]
    
    all_results = {}
    
    for dataset_name, dataset_path in datasets_config:
        result = test_on_dataset(model, dataset_name, dataset_path, device, args, config)
        all_results[dataset_name] = result
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    metrics = ['r_sum', 'txt_r1', 'txt_r5', 'txt_r10', 'txt_r_mean', 
               'img_r1', 'img_r5', 'img_r10', 'img_r_mean', 'r_mean']
    
    header = f"{'Dataset':<15}"
    for metric in metrics:
        header += f"{metric:<12}"
    print(header)
    print("-" * (15 + 12 * len(metrics)))
    
    for dataset_name, result in all_results.items():
        row = f"{dataset_name:<15}"
        for metric in metrics:
            value = result.get(metric, 0)
            if isinstance(value, float):
                row += f"{value:.2f}".ljust(12)
            else:
                row += f"{value}".ljust(12)
        print(row)
    
    print("="*80)
    
    result_file = "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_07/lama_test_results.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")


def parser_args():
    parser = argparse.ArgumentParser(description="Test on LAMA datasets")
    parser.add_argument('--config', type=str, default='./configs/vitb32/flickr/cusa.yaml', 
                        help='The config file. Default is the base cusa.yaml config.')
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_07/CLIP_ViT-B32_finetune_ALL_LoraRank16_seed0.ckpt',
                        help='The checkpoint file.')
    parser.add_argument('--gpu', default='cuda:0', type=str, help='GPU to use.')
    parser.add_argument("--num_workers", default=8, type=int, help="The number of workers for data loading.")
    parser.add_argument('--seed', default=23, type=int, help='Seed for testing.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser_args()
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # 确保 batch size 设置正确
    config['batch_size_testall'] = args.batch_size
    
    main(args, config)
