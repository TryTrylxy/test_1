# python check_compose.py --eval --checkpoint "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/output/vitb32/vg/cusa_restate_sl_seed23/checkpoints_2/checkpoint_best.pth" --config "./configs/vitb32/flickr/cusa_restate_sl_test_sl.yaml"

import os
from unire.model import unire
import torch
import yaml
import argparse
import json
import clip
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

def main():
    data_path = [
        # ["/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_03/dataset/my_f30k/new_test.json", "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/dataset/test/f30k_test_img_change/", 5, None],
        # ["/home/jncsnlp3/SSD2/syy/VL-check/data_cusa/Relation/vg/action.json"],
        # ["/home/jncsnlp3/SSD2/syy/VL-check/data_cusa/Relation/vg/spatial.json"],
        # ["/home/jncsnlp3/SSD2/syy/instruct-pix2pix-main/new_test.json", 1000],
        ["/home/jncsnlp3/SSD2/syy/lama-main/new_test.json", 1000],
    ]

    args = parser_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        config['save_path'] = config['save_path'] + "_seed" + str(args.seed)
        config['logger_name'] = os.path.join(config['save_path'], "log")
        config['model_name'] = os.path.join(config['save_path'], "checkpoints")
    args.gpu = "cuda:0"
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    model = unire(args, config)
    msg = model.load_state_dict(state_dict)
    print(msg)
    device = torch.device(args.gpu)
    model.to(device)
    for i in range(len(data_path)):
        test(data_path[i][0], model, device, data_path[i][1])

def test(data_path, model, device, num_data):
    dataset = test_dataset(data_path, model.preprocess, num_data)
    data_loader = create_loader(dataset, 128, 16)

    with torch.no_grad():
        model.eval()
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_embeds = []
        for i in tqdm(range(0, num_text, text_bs)):
            text = texts[i: min(num_text, i + text_bs)]
            text_input = data_loader.dataset.preprocess_text(text).to(device)
            text_embed = model.encode_text(text_input)
            text_embeds.append(text_embed)
        text_embeds = torch.cat(text_embeds, dim=0)
        
        texts = data_loader.dataset.text_neg
        num_text = len(texts)
        text_bs = 256
        text_embeds_neg = []
        for i in tqdm(range(0, num_text, text_bs)):
            text = texts[i: min(num_text, i + text_bs)]
            text_input = data_loader.dataset.preprocess_text(text).to(device)
            text_embed = model.encode_text(text_input)
            text_embeds_neg.append(text_embed)
        text_embeds_neg = torch.cat(text_embeds_neg, dim=0)

        image_embeds = []
        image_neg_embeds = []
        for image, image_neg, index in tqdm(data_loader):
            image = image.to(device)
            image_embed = model.encode_image(image)
            image_embeds.append(image_embed)

            image_neg = image_neg.to(device)
            image_neg_embed = model.encode_image(image_neg)
            image_neg_embeds.append(image_neg_embed)
        image_embeds = torch.cat(image_embeds, dim=0)
        image_neg_embeds = torch.cat(image_neg_embeds, dim=0)

        sims = get_similarity(image_embeds, text_embeds, data_loader.dataset.i2t_label)
        sims_neg = get_similarity(image_embeds, text_embeds_neg, data_loader.dataset.i2t_label)
        count = (sims_neg <= sims).sum().item()
        print("txt neg acc:", count / len(sims))
        
        sims = get_similarity(text_embeds, image_embeds, data_loader.dataset.t2i_label)
        sims_neg = get_similarity(text_embeds, image_neg_embeds, data_loader.dataset.t2i_label)
        count = (sims_neg <= sims).sum().item()
        print("img neg acc:", count / len(sims))

        score_matrix_i2t, score_matrix_t2i = model.get_similarity(
            image_embeds, text_embeds)
        sorted_values, sorted_indices = torch.sort(score_matrix_i2t, dim=1, descending=True)
        ranks = [torch.where(torch.isin(sorted_indices[i], torch.tensor(data_loader.dataset.i2t_label[i], device=device)))[0][0] for i in range(len(sorted_indices))]
        counts1 = [len([x for x in ranks if x < k]) / len(ranks) * 100 for k in [1, 5, 10, 50]]
        counts1 = [round(x, 2) for x in counts1]
        sorted_values, sorted_indices = torch.sort(score_matrix_t2i, dim=1, descending=True)
        ranks = [torch.where(torch.isin(sorted_indices[i], torch.tensor(data_loader.dataset.t2i_label[i], device=device)))[0][0] for i in range(len(sorted_indices))]
        counts2 = [len([x for x in ranks if x < k]) / len(ranks) * 100 for k in [1, 5, 10, 50]]
        counts2 = [round(x, 2) for x in counts2]
        print(counts1 + counts2 + [round(sum(counts1 + counts2) / len(counts1 + counts2), 2)])
    
def get_similarity(A, B, label):
    A = A / A.norm(dim=1, keepdim=True)
    B = B / B.norm(dim=1, keepdim=True)
    similarity = [
        torch.mm(A[i:i+1], B[ids].T).squeeze(0)  # 1×len(ids) → len(ids)
        for i, ids in enumerate(label)
    ]
    return torch.cat(similarity, dim=0)

def create_loader(dataset, batch_size, num_worker):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
        shuffle=False,
    )
    return loader

class test_dataset(Dataset):
    def __init__(self, dataPath, transform, num_data):
        self.transform = transform
        self.dataPath = dataPath

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

def parser_args():
    parser = argparse.ArgumentParser(description="PyTorch Image Retrieval Training")
    parser.add_argument('--config', type=str, default='', help='The config file.')
    parser.add_argument('--eval', action='store_true', help='Is eval?')
    parser.add_argument('--experiment', action='store_true', help='Is experiment?')
    parser.add_argument('--resume', action='store_true', help='Is resume?')
    parser.add_argument('--seed', default=23, type=int, help='Seed for initializing training.')
    parser.add_argument("--num_workers", default=8, type=int, help="The number of workers to use for data loading.")
    parser.add_argument('--distributed', default=True, type=bool, help='Is distributed?')
    parser.add_argument('--checkpoint', type=str, default='', help='The checkpoint file to resume from.')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    return args

main()