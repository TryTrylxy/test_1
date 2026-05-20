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
import logging
from datetime import datetime

def main():
    # 测试 LAMA 的三个数据集：Attribute, Object, Relation
    # 每个数据集使用各自的 new_test.json，限制 1000 条数据
    data_path = [
        ["/home/jncsnlp3/SSD2/syy/lama-main/dataset/Attribute/new_test.json", 1000],
        ["/home/jncsnlp3/SSD2/syy/lama-main/dataset/Object/new_test.json", 1000],
        ["/home/jncsnlp3/SSD2/syy/lama-main/dataset/Relation/new_test.json", 1000],
    ]
    dataset_names = ["Attribute", "Object", "Relation"]

    args = parser_args()
    
    # 设置随机种子以保证结果可复现
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 配置日志记录
    log_dir = "./test_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_log_{timestamp}.txt")
    
    # 创建日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Test started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Log file: {log_file}")
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        config['save_path'] = config['save_path'] + "_seed" + str(args.seed)
        config['logger_name'] = os.path.join(config['save_path'], "log")
        config['model_name'] = os.path.join(config['save_path'], "checkpoints")
    
    # 添加缺失的配置项（兼容 LoRA 模型）
    if 'do_mv' not in config:
        config['do_mv'] = False
    if 'do_neg' not in config:
        config['do_neg'] = True
    if 'do_restate_sl' not in config:
        config['do_restate_sl'] = False
    if 'do_test_sl' not in config:
        config['do_test_sl'] = False
    if 'do_LLMs_ab' not in config:
        config['do_LLMs_ab'] = False
    if 'do_uni_ab' not in config:
        config['do_uni_ab'] = False
    if 'do_cross_ab' not in config:
        config['do_cross_ab'] = False
    if 'is_all_gather' not in config:
        config['is_all_gather'] = False
    
    args.gpu = "cuda:0"
    
    # 加载 LoRA 模型权重
    logger.info(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    
    logger.info("Creating model...")
    model = unire(args, config)
    
    # LoRA 模型的权重键名与标准 CLIP 不同，使用 strict=False 加载
    # 这样可以忽略键名不匹配的部分，只加载能匹配的权重
    logger.info("Loading state dict (strict=False for LoRA compatibility)...")
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Model loaded: {msg}")
    
    device = torch.device(args.gpu)
    model.to(device)
    model.eval()
    
    # 在三个数据集上分别测试
    all_results = []
    for i in range(len(data_path)):
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing on {dataset_names[i]} dataset...")
        logger.info(f"{'='*80}")
        result = test(data_path[i][0], model, device, data_path[i][1], logger)
        all_results.append((dataset_names[i], result))
    
    # 打印汇总结果
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    header = f"{'Dataset':<15} {'txt_neg_acc':<15} {'img_neg_acc':<15} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'Mean':<8}"
    logger.info(header)
    logger.info(f"{'-'*80}")
    
    for name, result in all_results:
        txt_acc, img_acc, r1, r5, r10, mean = result
        row = f"{name:<15} {txt_acc:<15.2f} {img_acc:<15.2f} {r1:<8.2f} {r5:<8.2f} {r10:<8.2f} {mean:<8.2f}"
        logger.info(row)
    
    logger.info(f"{'='*80}")
    logger.info(f"Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results saved to: {log_file}")

def test(data_path, model, device, num_data, logger):
    """
    在单个数据集上测试模型性能
    
    Args:
        data_path: 数据集 JSON 文件路径
        model: 要测试的模型
        device: 设备 (cuda/cpu)
        num_data: 测试数据条数（默认 1000）
        logger: 日志记录器
    
    Returns:
        tuple: (txt_neg_acc, img_neg_acc, r1, r5, r10, mean)
    """
    dataset = test_dataset(data_path, model.preprocess, num_data)
    data_loader = create_loader(dataset, 128, 16)

    with torch.no_grad():
        model.eval()
        
        # 编码所有文本（正样本）
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_embeds = []
        for i in tqdm(range(0, num_text, text_bs), desc="Encoding text"):
            text = texts[i: min(num_text, i + text_bs)]
            text_input = data_loader.dataset.preprocess_text(text).to(device)
            text_embed = model.encode_text(text_input)
            text_embeds.append(text_embed)
        text_embeds = torch.cat(text_embeds, dim=0)
        
        # 编码所有文本（负样本）
        texts = data_loader.dataset.text_neg
        num_text = len(texts)
        text_bs = 256
        text_embeds_neg = []
        for i in tqdm(range(0, num_text, text_bs), desc="Encoding text neg"):
            text = texts[i: min(num_text, i + text_bs)]
            text_input = data_loader.dataset.preprocess_text(text).to(device)
            text_embed = model.encode_text(text_input)
            text_embeds_neg.append(text_embed)
        text_embeds_neg = torch.cat(text_embeds_neg, dim=0)

        # 编码所有图像（正样本和负样本）
        image_embeds = []
        image_neg_embeds = []
        for image, image_neg, index in tqdm(data_loader, desc="Encoding images"):
            image = image.to(device)
            image_embed = model.encode_image(image)
            image_embeds.append(image_embed)

            image_neg = image_neg.to(device)
            image_neg_embed = model.encode_image(image_neg)
            image_neg_embeds.append(image_neg_embed)
        image_embeds = torch.cat(image_embeds, dim=0)
        image_neg_embeds = torch.cat(image_neg_embeds, dim=0)

        # 计算负样本准确率（文本到图像）
        sims = get_similarity(image_embeds, text_embeds, data_loader.dataset.i2t_label)
        sims_neg = get_similarity(image_embeds, text_embeds_neg, data_loader.dataset.i2t_label)
        count = (sims_neg <= sims).sum().item()
        txt_neg_acc = count / len(sims) * 100
        logger.info(f"txt neg acc: {txt_neg_acc:.2f}%")
        
        # 计算负样本准确率（图像到文本）
        sims = get_similarity(text_embeds, image_embeds, data_loader.dataset.t2i_label)
        sims_neg = get_similarity(text_embeds, image_neg_embeds, data_loader.dataset.t2i_label)
        count = (sims_neg <= sims).sum().item()
        img_neg_acc = count / len(sims) * 100
        logger.info(f"img neg acc: {img_neg_acc:.2f}%")

        # 计算检索指标
        score_matrix_i2t, score_matrix_t2i = model.get_similarity(
            image_embeds, text_embeds)
        
        # Image-to-Text 检索指标 (只计算 R@1, R@5, R@10)
        sorted_values, sorted_indices = torch.sort(score_matrix_i2t, dim=1, descending=True)
        ranks = [torch.where(torch.isin(sorted_indices[i], torch.tensor(data_loader.dataset.i2t_label[i], device=device)))[0][0] for i in range(len(sorted_indices))]
        counts1 = [len([x for x in ranks if x < k]) / len(ranks) * 100 for k in [1, 5, 10]]
        counts1 = [round(x, 2) for x in counts1]
        
        # Text-to-Image 检索指标 (只计算 R@1, R@5, R@10)
        sorted_values, sorted_indices = torch.sort(score_matrix_t2i, dim=1, descending=True)
        ranks = [torch.where(torch.isin(sorted_indices[i], torch.tensor(data_loader.dataset.t2i_label[i], device=device)))[0][0] for i in range(len(sorted_indices))]
        counts2 = [len([x for x in ranks if x < k]) / len(ranks) * 100 for k in [1, 5, 10]]
        counts2 = [round(x, 2) for x in counts2]
        
        mean_acc = round(sum(counts1 + counts2) / len(counts1 + counts2), 2)
        
        logger.info(f"Image-to-Text: R@1={counts1[0]:.2f}, R@5={counts1[1]:.2f}, R@10={counts1[2]:.2f}")
        logger.info(f"Text-to-Image: R@1={counts2[0]:.2f}, R@5={counts2[1]:.2f}, R@10={counts2[2]:.2f}")
        logger.info(f"Mean: {mean_acc:.2f}")
        
        return txt_neg_acc, img_neg_acc, counts1[0], counts1[1], counts1[2], mean_acc
    
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